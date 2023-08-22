"""
The main class implementing the factorized prior model; see Ballé et al. 2017, "End-to-end optimized
image compression", and the tensorflow_compression implementation,
https://github.com/tensorflow/compression/blob/80d962f8f8532d9a3dbdaf0a97e249b7be7c29f6/models/bls2017.py
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
import mshyper
from mshyper.models import get_bottleneck_size, EMPTY_DICT
from common.lpips_tensorflow import learned_perceptual_metric_model

CODING_RANK = 3

# Hard-coded downsampling factor for the encoder (analysis + hyperanalysis).
DOWNSAMPLE_FACTOR = 16
MIN_IMAGE_DIM = 32  # lpips-vgg requires input at least this size.

# We use higher lambda at the beginning of training.
HIGHER_LAMBDA_UNTIL = 0.2
HIGHER_LAMBDA_FACTOR = 10.


# Encapsulates model + optimizer.
class Model(mshyper.models.Model):
  def __init__(self, scheduled_num_steps=1500000,
               rd_lambda=0.01,
               offset_heuristic=True,
               transform_config=EMPTY_DICT,
               optimizer_config=EMPTY_DICT, latent_config=immutabledict(uq=dict(method='unoise')),
               profile=False):
    super().__init__(scheduled_num_steps=scheduled_num_steps, rd_lambda=rd_lambda,
                     offset_heuristic=offset_heuristic, transform_config=transform_config,
                     optimizer_config=optimizer_config, latent_config=latent_config,
                     profile=profile)

  def _init_transforms(self, transform_config=EMPTY_DICT):
    analysis_cfg = dict(transform_config["analysis"])
    self._analysis = transform_builder.build(analysis_cfg.pop('cls'), **analysis_cfg)
    synthesis_cfg = dict(transform_config["synthesis"])
    self._synthesis = transform_builder.build(synthesis_cfg.pop('cls'), **synthesis_cfg)

    dummy_img_dim = MIN_IMAGE_DIM
    self._bottleneck_size = get_bottleneck_size(self._analysis, dummy_img_dim)

    self._prior = tfc.NoisyDeepFactorized(batch_shape=(self._bottleneck_size,))

    if self._profile:
      self._analysis = profile_utils.with_timing(tf.function(self._analysis))
      self._synthesis = profile_utils.with_timing(tf.function(self._synthesis))

      # Warm up tf.function on a dummy input to avoid inaccurate timing on the first call.
      dummy_img = tf.zeros([1, dummy_img_dim, dummy_img_dim, 3])
      _ = self.end_to_end_frame_loss(dummy_img, training=False)

  def infer_latent_rvs(self, x):
    """
    Inference path.
    :param x:
    :return:
    """
    x = pad_images(x, DOWNSAMPLE_FACTOR)  # TODO: obtain downsample factor from encoder.
    if self._profile:
      timing_info = dict()
      y, time = self._analysis(x)
      timing_info['analysis_time'] = time
    else:
      y = self._analysis(x)

    ret_val = LatentRVCollection(uq=(UQLatentRV(y),))
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

    entropy_model = tfc.ContinuousBatchedEntropyModel(
      self._prior, coding_rank=CODING_RANK, compression=False,
      offset_heuristic=self._offset_heuristic)

    if uq_method == "unoise":
      latent_sample, latent_bits = entropy_model(latent_rvs.uq[0].loc, training=training)
    elif uq_method == "mixedq":  # Mixed quantization.
      _, latent_bits = entropy_model(latent_rvs.uq[0].loc, training=training)
      latent_sample = entropy_model.quantize(latent_rvs.uq[0].loc)
    else:
      # Explicit sampling to support SGA itinf.
      reduce_axes = tuple(range(-CODING_RANK, 0))

      z_config = {**self.latent_config["uq"], "offset": entropy_model.quantization_offset}
      latent_sample = latent_rvs.uq[0].sample(training, **z_config)
      latent_bits = tf.reduce_sum(entropy_model.prior.log_prob(latent_sample),
                                  reduce_axes) / (
                      -tf.math.log(tf.constant(2, dtype=self._prior.dtype)))

    if self._profile:
      reconstruction, time = self._synthesis(latent_sample, training=training)
      timing_info['synthesis_time'] = time
    else:
      reconstruction = self._synthesis(latent_sample, training=training)
    reconstruction = unpad_images(reconstruction, image_batch.shape)

    bits = latent_bits  # [batchsize]

    num_pixels_per_image = tf.cast(tf.reduce_prod(tf.shape(image_batch)[1:-1]), bits.dtype)
    # batch_bpp = bits / num_pixels_per_image
    # bpp = tf.reduce_mean(batch_bpp)

    bpp = tf.reduce_mean(latent_bits) / num_pixels_per_image
    tf.debugging.check_numerics(bpp, "bpp")

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
