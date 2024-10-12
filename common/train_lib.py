"""
Helpers for model training and evaluation.
"""
import tensorflow as tf
from absl import logging
from typing import NamedTuple, Mapping, Any, Sequence
# from common import immutabledict
# from collections import OrderedDict
import ml_collections
import os
import shutil
import pprint
from clu import metric_writers, periodic_actions
from common.custom_writers import create_default_writer
import inspect
from pathlib import Path

from common.data_lib import get_dataset, get_dataset_from_glob
from common import utils


class Metrics(NamedTuple):
  """
  A container for metrics captured during training/evaluation; compatible with jitted tf code.
  Adapted from https://github.com/google-research/google-research/blob/master/vct/src/metric_collection.py
  """
  scalars: Mapping[str, Any]
  images: Mapping[str, tf.Tensor]

  # tensors: Mapping[str, tf.Tensor]
  @classmethod
  def make(cls):
    return Metrics(scalars={}, images={})

  def record_scalar(self, key, value):
    self.scalars[key] = value

  def record_scalars(self, scalars):
    # Inspired by writer_scalars https://github.com/google/CommonLoopUtils/blob/main/clu/metric_writers/summary_writer.py#L53
    for key, value in scalars.items():
      self.scalars[key] = value

  def record_image(self, key, value):
    self.images[key] = value

  @property
  def scalars_numpy(self):
    return {k: v.numpy().item() for (k, v) in self.scalars.items()}

  @property
  def images_grid(self):
    return {k: utils.visualize_image_batch(v, crop_to_max_dim=256) for (k, v) in self.images.items()}

  @property
  def scalars_float(self):
    return {k: float(v) for (k, v) in self.scalars.items()}

  @classmethod
  def merge_metrics(cls, metrics_list):
    """
    :param metrics_list: a list of Metrics, where each metrics object is the output of the model on an input batch.
    :return:
    """
    # Reduce scalars by taking means.
    merged_scalars = {}
    scalars_keys = metrics_list[0].scalars.keys()
    for key in scalars_keys:
      merged_scalars[key] = tf.reduce_mean([m.scalars[key] for m in metrics_list])

    # Reduce images/tensors by concatenating across batches.
    merged_images = {}
    images_keys = metrics_list[0].images.keys()
    for key in images_keys:
      merged_images[key] = tf.concat([m.images[key] for m in metrics_list], axis=0)

    return Metrics(scalars=merged_scalars, images=merged_images)


# Hardcoded identifiers.
TRAIN_COLLECTION = "train"
VAL_COLLECTION = "val"
CHECKPOINTS_DIR_NAME = "checkpoints"


# See https://github.com/google/flax/blob/e18a00a3b784afaf42825574836c5fe145688d8c/examples/ogbg_molpcba/train.py
# for example training loop with CLU. Perhaps also https://github.com/google/flax/blob/517b763590262d37fbbbd56ab262785cbbdb2c40/examples/imagenet/train.py
def simple_train_eval_loop(train_eval_config, workdir, model, train_dataset, val_data):
  """
  Template method for running the main training and evaluation loop.
  :param train_eval_config: a sub-dict of the main config, containing the following keys:
      - num_steps: an int, the number of training steps to run.
      - log_metrics_every_steps: an int, the number of training steps between logging metrics.
      - checkpoint_every_steps: an int, the number of training steps between saving checkpoints.
      - eval_every_steps: an int, the number of training steps between running validation.
      - warm_start: a string, the path to a checkpoint to optionally warm start from. For
        flexibility, this string can either be
        - a path to a checkpoint identifier,
        - or a path to a workdir whose 'train/checkpoints' subdir contains checkpoints,
        - or a path to an experiment dir (in which case the wid of the current job is used to load
          the checkpoint with the matching wid).
  :param workdir:
  :param model:
  :param train_dataset:
  :param val_data:
  :return:
  """
  logging.info("TF physical devices:\n%s", str(tf.config.list_physical_devices()))
  config = train_eval_config

  # Create writers for logs.
  train_dir = os.path.join(workdir, TRAIN_COLLECTION)
  # train_writer = metric_writers.create_default_writer(train_dir, collection=TRAIN_COLLECTION)
  train_writer = create_default_writer(train_dir)
  train_writer.write_hparams(config.to_dict())

  val_dir = os.path.join(workdir, VAL_COLLECTION)
  val_writer = create_default_writer(val_dir)

  checkpoint_dir = os.path.join(train_dir, CHECKPOINTS_DIR_NAME)
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  logging.info("Will save checkpoints to %s", checkpoint_dir)
  checkpoint = tf.train.Checkpoint(model=model)
  max_ckpts_to_keep = train_eval_config.get("max_ckpts_to_keep", 1)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir,
                                                  max_to_keep=max_ckpts_to_keep)

  if train_eval_config.get("warm_start"):
    warm_start = train_eval_config.warm_start

    def _locate_ckpt_path(warm_start: str):
      """
      Find the checkpoint path to warm start from.
      :param warm_start: either a path to a checkpoint identifier, or a path to a workdir
      whose 'train/checkpoints' subdir contains checkpoints, or a path to an experiment dir (in
      which case the wid of the current job is used to load the checkpoint with the matching wid).
      :return:
      """
      warm_start = Path(warm_start)
      if not warm_start.is_dir():
        raise ValueError

      # Check the provided dir.
      warm_start_dir = warm_start
      restore_ckpt_path = tf.train.latest_checkpoint(warm_start_dir)
      if restore_ckpt_path:
        return restore_ckpt_path

      # Treat it as a wu dir; check the train/checkpoints subdir.
      logging.info("No ckpt in warm_start dir; check its train subdir...")
      warm_start_dir = warm_start / TRAIN_COLLECTION / CHECKPOINTS_DIR_NAME
      restore_ckpt_path = tf.train.latest_checkpoint(warm_start_dir)
      if restore_ckpt_path:
        return restore_ckpt_path

      # Treat it as an experiment dir, and load the model with the matching wid as current run.
      logging.info("No ckpt so far; treat warm_start as experiment dir and check for matching work "
                   "unit id...")
      wid = utils.get_wid()
      assert wid is not None
      for wu_dir in warm_start.iterdir():
        if wu_dir.is_file() or "wid=" not in str(wu_dir):  # Skip things that aren't wu dirs.
          continue
        parsed_wid = utils.parse_runname(str(wu_dir), parse_numbers=False)["wid"]
        if wid == parsed_wid:
          warm_start_dir = wu_dir / TRAIN_COLLECTION / CHECKPOINTS_DIR_NAME
          break
      restore_ckpt_path = tf.train.latest_checkpoint(warm_start_dir)
      if restore_ckpt_path:
        return restore_ckpt_path

      if not restore_ckpt_path:
        raise ValueError()
      return None

    try:
      restore_ckpt_path = _locate_ckpt_path(warm_start)
      restore_status = checkpoint.restore(restore_ckpt_path)
      try:
        restore_status.assert_consumed()
      except:
        logging.warning("assert_consumed() failed...")
        restore_status.expect_partial()
      logging.info("Restored from %s", restore_ckpt_path)
    except ValueError:
      logging.warning(f"Failed to find ckpt from {warm_start}")
      checkpoint_manager.restore_or_initialize()

  else:
    checkpoint_manager.restore_or_initialize()

  initial_step = int(model.global_step.numpy())
  logging.info("Starting train eval loop at step %d.", initial_step)

  # Hooks called periodically during training.
  report_progress = periodic_actions.ReportProgress(
    num_train_steps=config.num_steps, writer=train_writer, every_secs=60)
  # profiler = periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
  hooks = [report_progress]

  train_iterator = iter(train_dataset)  # TODO: maybe just take train_iterator as the function arg.

  jit_compile = None  # True could be nice, but may not work.
  train_step_fn = tf.function(model.train_step, jit_compile=jit_compile, reduce_retracing=True)

  def train_fn(data_iterator):
    data_batch = next(data_iterator)
    metrics = train_step_fn(data_batch)
    return metrics

  val_step_fn = tf.function(model.validation_step)

  def evaluate_fn(step):
    metrics_list = []
    val_size = 0
    for data_batch in val_data:  # Assume finite!
      # metrics = model.validation_step(data_batch)
      metrics = val_step_fn(data_batch)
      metrics_list.append(metrics)
      val_size += data_batch.shape[0]

    metrics = Metrics.merge_metrics(metrics_list)

    val_writer.write_scalars(step, metrics.scalars_numpy)
    # val_writer.write_images(step, metrics.images_grid_np)
    val_writer.write_images(step, metrics.images_grid)
    logging.info("Ran validation on %d instances.", val_size)
    return None

  with metric_writers.ensure_flushes(train_writer):
    step = initial_step
    while step < config.num_steps:

      metrics = train_fn(train_iterator)
      if step % config.log_metrics_every_steps == 0:
        train_writer.write_scalars(step, metrics.scalars_float)
      for hook in hooks:
        hook(step)

      step += 1

      if (config.eval_every_steps > 0 and step % config.eval_every_steps == 0
            and step < config.num_steps):  # Will run final eval outside the training loop.
        logging.info("Evaluating at step %d", step)
        with report_progress.timed("eval"):
          evaluate_fn(step)

      if step % config.checkpoint_every_steps == 0 or step == config.num_steps:
        checkpoint_path = checkpoint_manager.save(step)
        logging.info("Saved checkpoint %s", checkpoint_path)

    logging.info("Finished training loop at step %d.", step)

    # Final evaluation.
    if config.eval_every_steps > 0:
      logging.info("Final eval outside of training loop.")
      with report_progress.timed("eval"):
        evaluate_fn(step)


def train_and_eval(config, model_cls, experiments_dir, runname):
  """
  Template method for launching training and evaluation of a model; basically sets up the model/data
  and does bookkeeping, then calls the simple_train_eval_loop method.
  :param config: a ml_collections.ConfigDict containing configurations. This is usually read from/
  defined in a config file. It should contain the following:
    - train_data_config: a dict with settings for training data containing the following keys:
      - dataset: a string, the name of the dataset to use.
      - batchsize: an int, the batchsize to use.
      - patchsize: an int, the patchsize to use.
    - val_data_config: a dict with settings for validation data containing the following keys:
      - dataset: a string, the name of the dataset to use.
      - batchsize: an int, the batchsize to use.
      - patchsize: an int, the patchsize to use.
      If unspecified, will use a random subset of the training data as validation data.
    - train_eval_config: a dict containing the following keys:
      - num_steps: an int, the number of training steps to run.
      - log_metrics_every_steps: an int, the number of training steps between logging metrics.
      - checkpoint_every_steps: an int, the number of training steps between saving checkpoints.
      - eval_every_steps: an int, the number of training steps between running validation.
      - warm_start: a string, the path to a checkpoint to optionally warm start from.
    - model_config: kwargs to pass to the model constructor.
  :param model_cls: the model class to use.
  :param experiments_dir: the pardir where experiments should be stored.
  :param runname: a (ideally unique) string that will be used to form the name of the workdir of
  this experiment.
  :return:
  """
  model_config = config["model_config"]
  model = model_cls(**model_config)

  def _get_dataset(data_config, split):
    dataset = get_dataset(data_config["dataset"], split=split, batchsize=data_config["batchsize"],
                          patchsize=data_config["patchsize"], normalize=True)
    return dataset

  train_dataset = _get_dataset(config["train_data_config"], "train")
  val_data_config = config.get("val_data_config", None)
  if val_data_config:
    val_dataset = _get_dataset(val_data_config, "validation")
  else:  # Will use this many random batches of train_dataset as validation data.
    VALIDATION_STEPS = 16
    val_dataset = train_dataset.take(VALIDATION_STEPS)

  ##################### BEGIN: Good old bookkeeping #########################
  xid = utils.get_xid()
  # Here, each runname is associated with a different work unit (Slurm call this a 'array job task')
  # within the same experiment. We add the work unit id prefix to make it easier to warm start
  # with the matching wid later.
  wid = utils.get_wid()
  if wid is None:
    wid_prefix = ''
  else:
    wid_prefix = f'wid={wid}-'
  workdir = os.path.join(experiments_dir, xid, wid_prefix + runname)
  # e.g., 'train_xms/21965/wid=3-mshyper-rd_lambda=0.08-latent_ch=320-base_ch=192'
  if not os.path.exists(workdir):
    os.makedirs(workdir)
  # absl logs from this point on will be saved to files in workdir.
  logging.get_absl_handler().use_absl_log_file(program_name="trainer", log_dir=workdir)

  logging.info("Using workdir:\n%s", workdir)
  logging.info("Input config:\n%s", pprint.pformat(config))

  # Save the config provided.
  with open(os.path.join(workdir, f"config.json"), "w") as f:
    f.write(config.to_json(indent=2))
  if "config_filename" in config:
    shutil.copy2(config["config_filename"], os.path.join(experiments_dir, xid, "config_script.py"))

  # Log more info.
  utils.log_run_info(workdir=workdir)
  # Write a copy of model source code.
  model_source_str = inspect.getsource(inspect.getmodule(model_cls))
  with open(os.path.join(workdir, f"models.py"), "w") as f:
    f.write(model_source_str)
  ##################### END: Good old bookkeeping #########################

  train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
  val_data = val_dataset.cache()

  # Run custom training loop.
  return simple_train_eval_loop(config.train_eval_config, workdir, model, train_dataset, val_data)
