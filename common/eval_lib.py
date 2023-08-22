import tensorflow as tf
from pathlib import Path
from absl import logging
import imp
from common.train_lib import TRAIN_COLLECTION, CHECKPOINTS_DIR_NAME
from common.utils import EMPTY_DICT, load_json, dump_json, parse_runname
import re
import pprint


def load_latest_ckpt(workdir, model_cls=None, load_model_config=True,
                     update_model_config=EMPTY_DICT, verbose=False):
  """
  Helper function to load the latest model ckpt from a given workdir.
  :param workdir: e.g., 'train_xms/21965/mshyper-rd_lambda=0.08-latent_ch=320-base_ch=192'.
  :param model_cls: if None, will use the 'models.py' saved in the workdir, otherwise will use the
  given model_cls to instantiate the model.
  :param load_model_config: if True, will use the config saved in the workdir.
  :param update_model_config: a dict to specify custom model_config.
  :return:
  """
  workdir = Path(workdir)

  if model_cls is None:
    model_src_path = workdir / "models.py"
    models_module = imp.load_source("models", str(model_src_path))
    model_cls = models_module.Model

  if load_model_config:
    cfg_path = workdir / "config.json"
    config = load_json(cfg_path)
    model_config = config["model_config"]
  else:
    model_config = {}

  model_config.update(update_model_config)
  model = model_cls(**model_config)
  if verbose:
    logging.info("Instantiated model with the following model_config: \n%s", pprint.pformat((
      model_config)))

  checkpoint_dir = workdir / TRAIN_COLLECTION / CHECKPOINTS_DIR_NAME
  checkpoint = tf.train.Checkpoint(model=model)
  ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
  restore_status = checkpoint.restore(ckpt_path)
  try:
    restore_status.assert_consumed()
  except:
    logging.warning("assert_consumed() failed...")
    restore_status.expect_partial()

  logging.info("Restored model from %s", ckpt_path)
  return model


def eval_workdir(workdir, eval_data, results_dir, model_cls=None, profile=False,
                 skip_existing=True):
  """
  Load the latest model ckpt from a given workdir, and evaluate on eval_data by calling
  model.evaluate().
  :param workdir:
  :param eval_data:
  :param results_dir:
  :param model_cls:
  :param skip_existing: if True, will skip the evaluation if a results file with the same name
    already exists.
  :return: results_file_path:
  """
  if not tf.io.gfile.exists(results_dir):
    tf.io.gfile.makedirs(results_dir)
    logging.info(f"Created {results_dir} since it doesn't exist")

  # e.g., 'train_xms/21965/wid=3-mshyper-rd_lambda=0.08-latent_ch=320-base_ch=192'.
  # See common.train_lib for details on how workdir is generated.
  workdir = Path(workdir)
  runname = workdir.name
  runname = re.sub(r"^wid=\d+-", "", runname)  # 'mshyper-bpp_c=0.15-latent_ch=320-base_ch=192'
  xid = workdir.parent.name  # '21965'

  if profile:
    model = load_latest_ckpt(workdir, model_cls=model_cls, update_model_config={'profile': True})
  else:
    model = load_latest_ckpt(workdir, model_cls=model_cls)
  model_step = int(model.global_step)
  results_file_name = f"{runname}-step={model_step:3g}-xid={xid}.json"
  results_file_path = Path(results_dir) / results_file_name
  if tf.io.gfile.exists(results_file_path) and skip_existing:
    logging.info(f"Skipping existing results file {results_file_path}")
    return results_file_path

  metrics_list = list(model.evaluate(eval_data))
  results_metrics_list = [metric.scalars_float for metric in
                          metrics_list]  # Will be a flat list of dicts

  # Extract hyparameters from runname and add to each dict in results_metrics_list. Useful for
  # parsing results later.
  runname_hparams = parse_runname(runname, parse_numbers=True)
  for instance_id, metrics_dict in enumerate(results_metrics_list):
    metrics_dict['instance_id'] = instance_id
    metrics_dict.update(runname_hparams)

  dump_json(results_metrics_list, results_file_path)
  logging.info(f'Saved results to {results_file_path}')

  return results_file_path
