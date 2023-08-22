"""
Helper routines for running iterative inference (e.g., SGA) on image batches.
"""
import tensorflow as tf
import numpy as np
from absl import logging
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

TRAIN_COLLECTION = "train"
VAL_COLLECTION = "val"
CHECKPOINTS_DIR_NAME = "checkpoints"


def itinf_on_data_batch(train_eval_config, train_writer, val_writer, model, data_batch):
  """
  Template method for running the main iterative inference loop on a single data batch.
  Similar to the training loop in train_lib.simple_train_eval_loop.
  :param train_eval_config:
  :param train_writer:
  :param val_writer:
  :param model: should implement `itinf_train_step` and `itinf_validation_step` methods.
  :param data_batch:
  :return:
  """
  config = train_eval_config
  # initial_step = int(model.global_step.numpy())
  initial_step = 0
  logging.info("Starting itinf optimization loop at step %d.", initial_step)

  # Hooks called periodically during training.
  report_progress = periodic_actions.ReportProgress(
    num_train_steps=config.num_steps, writer=train_writer, every_secs=60)
  # profiler = periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
  hooks = [report_progress]

  train_step_fn = tf.function(model.itinf_train_step)  # Force retracing.
  # train_step_fn = model.itinf_train_step  # Eager mode for debugging.

  train_metrics = []
  val_metrics = []

  def evaluate_fn(step):
    metrics = model.itinf_validation_step(data_batch, training=False)

    val_writer.write_scalars(step, metrics.scalars_numpy)
    val_writer.write_images(step, metrics.images_grid)
    val_metrics.append({"step": step, **metrics.scalars_float})
    return None

  model.initialize_itinf(data_batch)
  logging.info("Created model.latent_rvs: %s", str(model.latent_rvs))

  with metric_writers.ensure_flushes(train_writer):
    step = initial_step
    while step < config.num_steps:
      metrics = train_step_fn(data_batch)
      if step % config.log_metrics_every_steps == 0:
        train_writer.write_scalars(step, metrics.scalars_float)
        train_metrics.append({"step": step, **metrics.scalars_float})
      for hook in hooks:
        hook(step)

      step += 1

      if (config.eval_every_steps > 0 and step % config.eval_every_steps == 0
            and step < config.num_steps):  # Will run final eval outside the training loop.
        logging.info("Evaluating at step %d", step)
        with report_progress.timed("eval"):
          evaluate_fn(step)

    logging.info("Finished training loop at step %d.", step)

    # Final validation.
    if config.eval_every_steps > 0:
      logging.info("Final eval outside of training loop.")
      with report_progress.timed("eval"):
        evaluate_fn(step)

    # Save the final state of the variables that have been optimized as npy arrays.
    itinf_vars_npy_dict = {v.name: v.numpy() for v in model.itinf_trainable_variables}
  return train_metrics, val_metrics, itinf_vars_npy_dict


from common.eval_lib import load_latest_ckpt


def itinf_eval(config, model_cls, experiments_dir, runname):
  """
  Template method for launching iterative inference on a dataset at test/eval time, similar to the
  train_and_eval method in train_lib.py. We load the checkpoint of a pretrained model specified by
  configs and run iterative inference on the given dataset; the results are saved to the workdir
  of the itinf experiment.
  :param config: a ml_collections.ConfigDict. This is similar to the config used in model training,
  except that:
  - there is a data_config key (instead of train_data_config and val_data_config) to specify
    the dataset to run/eval iterative inference on;
  - the train_eval_config will refer to the iterative inference optimization; also, unlike the
    configs used in model training, the train_eval_config should contain a warm_start_exp_dir and
    warm_start_wid to specify the model checkpoint to load (this is so that we can launch multiple
    itinf experiments in parallel, each with a different warm start model checkpoint, usually
    corresponding to a different hyperparameter setting (e.g., lambda) used in training);
  - any optimizer config used with model_config will specify the iterative inference optimizer;
  - model_config should also contain a 'latent_config' specifying the type of latent variables and
    how they will be optimized (e.g., SGA, with corresponding temperature schedule)

  :param model_cls: the model class to use.
  :param experiments_dir: the pardir where experiments should be stored.
  :param runname: a (ideally unique) string that will be used to form the name of the workdir of
  this experiment.
  :return:
  """
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

  # Set up data.
  data_config = config["data_config"]
  dataset = get_dataset(data_config["dataset"], split="test", batchsize=data_config["batchsize"],
                        patchsize=data_config["patchsize"], normalize=True)

  # Create / load model.
  warm_start_exp_dir = Path(config['train_eval_config']['warm_start_exp_dir'])
  wwid = int(config['train_eval_config']['warm_start_wid'])
  # Below adapted from train_lib.simple_train_eval_loop
  for wu_dir in warm_start_exp_dir.iterdir():
    if wu_dir.is_file() or "wid=" not in str(wu_dir):  # Skip things that aren't wu dirs.
      continue
    parsed_wid = utils.parse_runname(str(wu_dir), parse_numbers=False)["wid"]
    if wwid == int(parsed_wid):
      break
  else:
    raise ValueError(f"No wid matches given warm_start_wid={wwid}")
  model = load_latest_ckpt(workdir=wu_dir, model_cls=model_cls, load_model_config=True,
                           update_model_config=config['model_config'], verbose=True)

  train_eval_config = config['train_eval_config']
  # See https://github.com/google/flax/blob/e18a00a3b784afaf42825574836c5fe145688d8c/examples/ogbg_molpcba/train.py
  # for example training loop with CLU. Perhaps also https://github.com/google/flax/blob/517b763590262d37fbbbd56ab262785cbbdb2c40/examples/imagenet/train.py
  # return simple_train_eval_loop(config.train_eval_config, workdir, model, dataset)

  logging.info("TF physical devices:\n%s", str(tf.config.list_physical_devices()))
  # For distributed training, may want to instantiate model within this method, by accepting create_model_fn.

  # Loop through each data batch and run a separate itinf optimization.
  for batch_id, data_batch in enumerate(dataset):
    batch_workdir = os.path.join(workdir, f"batch_id={batch_id}")
    logging.info("Starting batch %d in %s", batch_id, batch_workdir)
    # Create writers for logs.
    train_dir = os.path.join(batch_workdir, TRAIN_COLLECTION)
    # train_writer = metric_writers.create_default_writer(train_dir, collection=TRAIN_COLLECTION)
    train_writer = create_default_writer(train_dir)
    train_writer.write_hparams(train_eval_config.to_dict())

    val_dir = os.path.join(batch_workdir, VAL_COLLECTION)
    val_writer = create_default_writer(val_dir)

    train_metrics, val_metrics, itinf_vars_npy_dict = itinf_on_data_batch(train_eval_config,
                                                                          train_writer, val_writer,
                                                                          model, data_batch)
    train_metrics = [{"batch_id": batch_id, **d} for d in train_metrics]
    val_metrics = [{"batch_id": batch_id, **d} for d in val_metrics]

    utils.dump_json(train_metrics, os.path.join(train_dir, "metrics.json"))
    utils.dump_json(val_metrics, os.path.join(val_dir, "metrics.json"))
    np.savez_compressed(os.path.join(batch_workdir, "itinf_vars.npz"), **itinf_vars_npy_dict)

    logging.info("Done with batch %d.", batch_id)
