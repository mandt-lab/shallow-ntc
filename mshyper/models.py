"""
The main class implementing the mean-scale hyperprior model; see Minnen et al. 2018, "Joint
Autoregressive and Hierarchical Priors for Learned Image Compression", as well as
https://github.com/tensorflow/compression/blob/80d962f8f8532d9a3dbdaf0a97e249b7be7c29f6/models/bmshj2018.py
The various transforms are implemented in common/transforms.py, and can be specified via strings
in the transform_config kwarg of the model constructor.
"""
import tensorflow as tf
import tensorflow_compression as tfc
from common.transforms import class_builder as transform_builder  # Just use the default class map.
from common.utils import ClassBuilder
from common.latent_rvs_lib import UQLatentRV, LatentRVCollection
from common.latent_rvs_utils import sga_schedule_at_step
from common.immutabledict import immutabledict
from common.image_utils import mse_psnr, pad_images, unpad_images
from common import data_lib
from common import schedule
from common import profile_utils
from common.train_lib import Metrics
from collections import OrderedDict
from ml_collections import ConfigDict
from absl import logging
from common.lpips_tensorflow import learned_perceptual_metric_model

EMPTY_DICT = immutabledict()

# Fixed configs for the ScaleIndexedEntropyModel.
NUM_SCALES = 64
SCALE_MIN = 0.11
SCALE_MAX = 256.
SCALE_FACTOR = (tf.math.log(SCALE_MAX) - tf.math.log(SCALE_MIN)) / (NUM_SCALES - 1.)
SCALE_FN = lambda i: tf.math.exp(tf.math.log(SCALE_MIN) + SCALE_FACTOR * i)

CODING_RANK = 3

# Dummy image for initialization etc.; should be >= the downsample factor of the model.
DUMMY_IMG_DIM = 64

# We use higher lambda at the beginning of training.
HIGHER_LAMBDA_UNTIL = 0.2
HIGHER_LAMBDA_FACTOR = 10.


# Encapsulates model + optimizer.
class Model(tf.Module):
  def __init__(self, scheduled_num_steps=1500000,
               rd_lambda=0.01,
               offset_heuristic=True,
               transform_config=EMPTY_DICT,
               optimizer_config=EMPTY_DICT, latent_config=immutabledict(uq=dict(method='unoise')),
               profile=False):
    """
    Instantiates the model and optimizer.
    :param scheduled_num_steps: total number of training steps (mostly used for setting various
      schedules, e.g., learning rate schedule).
    :param rd_lambda:
    :param offset_heuristic:
    :param transform_config:
    :param optimizer_config:
    :param latent_config:
    :param profile: whether to turn on profiling for the transforms and report timing info.
    """
    super().__init__()
    self._scheduled_num_steps = scheduled_num_steps
    self._rd_lambda = rd_lambda

    self._latent_config = latent_config

    uq_method = self._latent_config["uq"].get("method", "unoise")
    if uq_method == "mixedq" and offset_heuristic:
      offset_heuristic = False
      print("Warning: modifying offset_heuristic from True to False, as it doesn't make sense for"
            "mixedq training.")  # TODO: make less ugly.
      logging.warning("modifying offset_heuristic from True to False, as it doesn't make sense for"
                      "mixedq training.")
    self._offset_heuristic = offset_heuristic

    # Flag indicating whether the model is in iterative inference mode.
    self.itinf = False

    # Set up lr and optimizer
    self._optimizer_config = optimizer_config
    optimizer, lr_schedule_fn = self._get_optimizer(self._optimizer_config,
                                                    self._scheduled_num_steps)
    self.optimizer = optimizer
    self._lr_schedule_fn = lr_schedule_fn

    self._transform_config = transform_config
    self._profile = profile
    self._init_transforms(transform_config)

  def _get_optimizer(self, optimizer_config, scheduled_num_steps):
    optimizer_config = dict(optimizer_config)  # Make a copy to avoid mutating the original.

    learning_rate = optimizer_config.pop("learning_rate", 1e-4)
    reduce_lr_after = optimizer_config.pop("reduce_lr_after", 0.8)
    reduce_lr_factor = optimizer_config.pop("reduce_lr_factor", 0.1)
    if "warmup_steps" in optimizer_config:
      warmup_steps = optimizer_config.pop("warmup_steps")
    else:
      warmup_until = optimizer_config.pop("warmup_until", 0.02)
      warmup_steps = int(warmup_until * scheduled_num_steps)
    lr_schedule_fn = schedule.CompressionSchedule(base_learning_rate=learning_rate,
                                                  total_num_steps=scheduled_num_steps,
                                                  warmup_steps=warmup_steps,
                                                  drop_after=reduce_lr_after,
                                                  drop_factor=reduce_lr_factor)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_fn, **optimizer_config)
    return optimizer, lr_schedule_fn

  def _init_transforms(self, transform_config=EMPTY_DICT):
    analysis_cfg = dict(transform_config["analysis"])
    self._analysis = transform_builder.build(analysis_cfg.pop('cls'), **analysis_cfg)
    synthesis_cfg = dict(transform_config["synthesis"])
    self._synthesis = transform_builder.build(synthesis_cfg.pop('cls'), **synthesis_cfg)

    dummy_img = tf.zeros([1, DUMMY_IMG_DIM, DUMMY_IMG_DIM, 3])
    dummy_latents = self._analysis(dummy_img)
    self._bottleneck_size = bottleneck_size = dummy_latents.shape[-1]
    if "hyper_analysis" in transform_config:
      hyper_analysis_cfg = dict(transform_config["hyper_analysis"])
    else:
      hyper_analysis_cfg = dict(cls='HyperAnalysis', bottleneck_size=bottleneck_size)
    self._hyper_analysis = transform_builder.build(hyper_analysis_cfg.pop('cls'),
                                                   **hyper_analysis_cfg)
    if "hyper_synthesis" in transform_config:
      hyper_synthesis_cfg = dict(transform_config["hyper_synthesis"])
    else:
      hyper_synthesis_cfg = dict(cls='HyperSynthesis', bottleneck_size=bottleneck_size)
    self._hyper_synthesis = transform_builder.build(hyper_synthesis_cfg.pop('cls'),
                                                    **hyper_synthesis_cfg)

    dummy_hyper_latents = self._hyper_analysis(dummy_latents)
    hyper_bottleneck_size = dummy_hyper_latents.shape[-1]
    self._prior = tfc.NoisyDeepFactorized(batch_shape=(hyper_bottleneck_size,))

    dummy_hyper_latents_dim = dummy_hyper_latents.shape[-2]
    self.downsample_factor = int(DUMMY_IMG_DIM / dummy_hyper_latents_dim)
    assert dummy_hyper_latents_dim * self.downsample_factor == DUMMY_IMG_DIM, "Downsample "
    "factor should divide evenly into the dimension of init dummy image."

    if self._profile:
      self._analysis = profile_utils.with_timing(tf.function(self._analysis))
      self._synthesis = profile_utils.with_timing(tf.function(self._synthesis))
      self._hyper_analysis = profile_utils.with_timing(tf.function(self._hyper_analysis))
      self._hyper_synthesis = profile_utils.with_timing(tf.function(self._hyper_synthesis))

      # Warm up tf.function on a dummy input to avoid inaccurate timing on the first call.
      _ = self.end_to_end_frame_loss(dummy_img, training=False)

  @property
  def global_step(self):
    if self.itinf:
      return self.itinf_optimizer.iterations
    else:
      return self.optimizer.iterations

  @property
  def _scheduled_lr(self):
    # This is just for logging/debugging purpose. Should equal self._lr_schedule_fun(self.global_step)
    # Also see https://github.com/google-research/google-research/blob/bb5e979a2d9389850fda7eb837ef9c8b8ba8244b/vct/src/models.py#672
    if self.itinf:
      optimizer = self.itinf_optimizer
    else:
      optimizer = self.optimizer
    return optimizer._decayed_lr(tf.float32)

  @property
  def _scheduled_rd_lambda(self):
    """Returns the scheduled rd-lambda.
    Based on https://github.com/google-research/google-research/blob/master/vct/src/models.py#L400
    """
    _rd_lambda = tf.convert_to_tensor(self._rd_lambda)
    if self._rd_lambda <= 0.01 and not self.itinf:  # Only do lambda warmup during model training.
      schedule_value = schedule.schedule_at_step(
        self.global_step,
        vals=[HIGHER_LAMBDA_FACTOR, 1.],
        boundaries=[int(self._scheduled_num_steps * HIGHER_LAMBDA_UNTIL)],
        interpolation=schedule.InterpolationType.CONSTANT
      )
      schedule_value = _rd_lambda * schedule_value
    else:
      schedule_value = _rd_lambda
    return schedule_value

  @property
  def latent_config(self):
    """
    Return a copy of self._latent_config, but with some attributes set dynamically based on
    global_step (e.g., annealing temperature).
    :return:
    """
    if isinstance(self._latent_config, ConfigDict):
      config = self._latent_config.to_dict()
    elif isinstance(self._latent_config, immutabledict):
      config = dict(self._latent_config)
    else:
      assert isinstance(self._latent_config, dict)
      config = self._latent_config.copy()

    if 'uq' in config:
      cfg = config['uq']
      if cfg['method'] == 'sga':
        tau = sga_schedule_at_step(self.global_step, r=cfg['tau_r'], ub=cfg['tau_ub'],
                                   lb=cfg.get('tau_lb', 1e-8), t0=cfg['tau_t0'])
        cfg['tau'] = tau
    # else:  # If no uq method was specified, we default to 'unoise' for training.
    #   assert not self.itinf
    #   config['uq'] = dict(method='unoise')
    return config

  def infer_latent_rvs(self, x):
    """
    Inference path.
    :param x:
    :return:
    """
    x = pad_images(x, self.downsample_factor)
    if self._profile:
      timing_info = dict()
      y, time = self._analysis(x)
      timing_info['analysis_time'] = time
      z, time = self._hyper_analysis(y)
      timing_info['hyper_analysis_time'] = time
    else:
      y = self._analysis(x)
      z = self._hyper_analysis(y)

    ret_val = LatentRVCollection(uq=(UQLatentRV(z), UQLatentRV(y)))
    if self._profile:
      ret_val = (ret_val, timing_info)
    return ret_val

  def frame_loss_given_latent_rvs(self, image_batch, latent_rvs, training):
    """
    Generative path + losses.
    :param image_batch:
    :param latent_rvs:
    :return:
    """
    uq_method = self._latent_config["uq"].get("method", "unoise")

    if self._profile:
      timing_info = {}

    entropy_model = tfc.LocationScaleIndexedEntropyModel(
      tfc.NoisyNormal, NUM_SCALES, SCALE_FN, coding_rank=CODING_RANK,
      compression=False)
    side_entropy_model = tfc.ContinuousBatchedEntropyModel(self._prior, coding_rank=CODING_RANK,
                                                           compression=False,
                                                           offset_heuristic=self._offset_heuristic)

    if uq_method == "unoise":
      hyper_latent_sample, hyper_latent_bits = side_entropy_model(latent_rvs.uq[0].loc,
                                                                  training=training)
    elif uq_method == "mixedq":  # Mixed quantization.
      _, hyper_latent_bits = side_entropy_model(latent_rvs.uq[0].loc,
                                                training=training)
      hyper_latent_sample = side_entropy_model.quantize(latent_rvs.uq[0].loc)
    else:
      # Explicit sampling to support SGA itinf.
      reduce_axes = tuple(range(-CODING_RANK, 0))

      z_config = {**self.latent_config["uq"], "offset": side_entropy_model.quantization_offset}
      hyper_latent_sample = latent_rvs.uq[0].sample(training, **z_config)
      hyper_latent_bits = tf.reduce_sum(side_entropy_model.prior.log_prob(hyper_latent_sample),
                                        reduce_axes) / (
                            -tf.math.log(tf.constant(2, dtype=self._prior.dtype)))
    if self._profile:
      hyper_syn_res, time = self._hyper_synthesis(hyper_latent_sample)
      timing_info['hyper_synthesis_time'] = time
    else:
      hyper_syn_res = self._hyper_synthesis(hyper_latent_sample)
    mu, sigma = tf.split(hyper_syn_res, num_or_size_splits=2, axis=-1)
    sigma = tf.exp(sigma)  # make positive; will be clipped then quantized to scale_table anyway
    loc, indexes = mu, sigma
    if uq_method == "unoise":  # Being lazy here, and defaulting unspecified uq method to unoise.
      latent_sample, latent_bits = entropy_model(latent_rvs.uq[1].loc, indexes, loc=loc,
                                                 training=training)
    elif uq_method == "mixedq":  # Mixed quantization.
      _, latent_bits = entropy_model(latent_rvs.uq[1].loc, indexes, loc=loc,
                                     training=training)
      latent_sample = entropy_model.quantize(latent_rvs.uq[1].loc, loc=loc)
    else:
      # Explicit sampling to support SGA itinf.
      y_config = {**self.latent_config["uq"], "offset": loc}
      latent_sample = latent_rvs.uq[1].sample(training, **y_config)
      py_centered = entropy_model._make_prior(entropy_model._normalize_indexes(indexes))  # loc=0 py
      # Important: need to center the latent_sample before evaluating it under py_centered.
      latent_bits = tf.reduce_sum(py_centered.log_prob(latent_sample - loc), reduce_axes) / (
        -tf.math.log(tf.constant(2, dtype=self._prior.dtype)))

    if self._profile:
      reconstruction, time = self._synthesis(latent_sample, training=training)
      timing_info['synthesis_time'] = time
    else:
      reconstruction = self._synthesis(latent_sample, training=training)
    reconstruction = unpad_images(reconstruction, image_batch.shape)

    bits = hyper_latent_bits + latent_bits  # [batchsize]

    num_pixels_per_image = tf.cast(tf.reduce_prod(tf.shape(image_batch)[1:-1]), bits.dtype)
    # batch_bpp = bits / num_pixels_per_image
    # bpp = tf.reduce_mean(batch_bpp)

    hyper_latent_bpp = tf.reduce_mean(hyper_latent_bits) / num_pixels_per_image
    latent_bpp = tf.reduce_mean(latent_bits) / num_pixels_per_image
    tf.debugging.check_numerics(hyper_latent_bpp, "hyper_latent_bpp")
    tf.debugging.check_numerics(latent_bpp, "latent_bpp")
    bpp = hyper_latent_bpp + latent_bpp

    # Covert to [0, 255] to compute distortion.
    image_batch = data_lib.floats_to_pixels(image_batch, training=training)
    reconstruction = data_lib.floats_to_pixels(reconstruction, training=training)
    batch_mse, batch_psnr = mse_psnr(image_batch, reconstruction)
    mse = tf.reduce_mean(batch_mse)
    psnr = tf.reduce_mean(batch_psnr)

    record_dict = {}
    # Compute MS-SSIM in validation mode.
    if not training:
      max_pxl_val = 255.
      im_size = tf.shape(image_batch)[1:-1]
      # tf.image.ssim_multiscale seems to crash when input < 160x160
      if im_size[0] < 160 and im_size[1] < 160:
        # TODO: provide warning
        batch_msssim = tf.image.ssim(image_batch, reconstruction, max_val=max_pxl_val)
      else:
        batch_msssim = tf.image.ssim_multiscale(image_batch, reconstruction, max_val=max_pxl_val)
      batch_msssim_db = -10. * tf.math.log(1 - batch_msssim) / tf.math.log(10.)
      record_dict["msssim"] = tf.reduce_mean(batch_msssim)
      record_dict["msssim_db"] = tf.reduce_mean(batch_msssim_db)

      if tf.executing_eagerly():  # Evaluate LPIPS when in eager mode.
        im_size = tuple(im_size.numpy())
        # tf.Tensor -> tuple so @cache can hash it ok (alternatively, disable caching behavior
        # of the lpips module to allow proper JITing in graph mode).
        lpips_model = learned_perceptual_metric_model(im_size)
        batch_lpips = lpips_model([image_batch, reconstruction])
        record_dict["lpips"] = tf.reduce_mean(batch_lpips)

    metrics = Metrics.make()
    rd_loss = bpp + self._scheduled_rd_lambda * mse
    metrics.record_scalar('sched_rd_lambda', self._scheduled_rd_lambda)

    if self.latent_config['uq']['method'] == 'sga':
      metrics.record_scalar('tau', self.latent_config['uq']['tau'])

    record_dict.update(
      dict(rd_loss=rd_loss, bpp=bpp, mse=mse, psnr=psnr, scheduled_lr=self._scheduled_lr))
    if self._profile:
      record_dict.update(timing_info)

    metrics.record_scalars(record_dict)
    # Check for NaNs in the loss
    tf.debugging.check_numerics(rd_loss, "rd_loss")

    metrics.record_image("reconstruction", reconstruction)
    return rd_loss, metrics

  def end_to_end_frame_loss(self, image_batch, training):
    if self._profile:
      latent_rvs, inf_timing = self.infer_latent_rvs(image_batch)
    else:
      latent_rvs = self.infer_latent_rvs(image_batch)

    loss, metrics = self.frame_loss_given_latent_rvs(image_batch, latent_rvs=latent_rvs,
                                                     training=training)

    if self._profile:  # Add timing info from inference path to metrics too, if in profile mode.
      metrics.record_scalars(inf_timing)

    return loss, metrics

  def train_step(self, image_batch):
    with tf.GradientTape() as tape:
      loss, metrics = self.end_to_end_frame_loss(image_batch, training=True)

    var_list = self.trainable_variables
    gradients = tape.gradient(loss, var_list)
    self.optimizer.apply_gradients(zip(gradients, var_list))

    return metrics

  def validation_step(self, image_batch, training=False) -> Metrics:
    loss, metrics = self.end_to_end_frame_loss(image_batch, training=training)
    return metrics

  def initialize_itinf(self, image_batch):
    self.latent_rvs = self.infer_latent_rvs(image_batch).get_trainable_copy()

    optimizer, lr_schedule_fn = self._get_optimizer(self._optimizer_config,
                                                    self._scheduled_num_steps)
    self.itinf_optimizer = optimizer
    self.itinf = True

  @property
  def itinf_trainable_variables(self):
    return self.latent_rvs.trainable_variables

  def itinf_train_step(self, image_batch):
    with tf.GradientTape() as tape:
      loss, metrics = self.frame_loss_given_latent_rvs(image_batch, latent_rvs=self.latent_rvs,
                                                       training=True)
    var_list = self.itinf_trainable_variables
    gradients = tape.gradient(loss, var_list)
    self.itinf_optimizer.apply_gradients(zip(gradients, var_list))
    return metrics

  def itinf_validation_step(self, image_batch, training=False) -> Metrics:
    loss, metrics = self.frame_loss_given_latent_rvs(image_batch, latent_rvs=self.latent_rvs,
                                                     training=training)
    return metrics

  def evaluate(self, images) -> Metrics:
    """
    Used for getting final results.
    If a [B, H, W, 3] tensor is provided, will evaluate on individual image
    tensors ([1, H, W, 3]) in order. Otherwise, we assume a caller has passed in
    an iterable of images (although we do not verify that each image tensor has
    batch size = 1).
    :param images:
    :return:
    """
    if isinstance(images, tf.Tensor):
      batch_size = images.shape[0]
      images = tf.split(images, batch_size)
    else:
      images = images

    for img in images:
      loss, metrics = self.end_to_end_frame_loss(img, training=False)
      yield metrics
