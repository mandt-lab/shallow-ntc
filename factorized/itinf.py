# A cmdline program for test-time iterative inference. Largely based on train.py.
# Example run (from project root dir):
# python -m factorized.itinf --config factorized/configs/itinf_rdl.py

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
from factorized.models import Model
from common.itinf_lib import itinf_eval

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None,
                                'File path to the training hyperparameter configuration.',
                                lock_config=True)
flags.DEFINE_integer('hid', None, "ID of the hyperparameter configuration to use as defined in"
                                  "the config script's get_hyper().")
flags.DEFINE_string('experiments_dir', './itinf_xms', 'Directory to store experiment data.')


# Some boilerplate.
def load_config_module():
  # Inspired by the source code for ml_collections/config_flags/config_flags._ConfigFileParser,
  # https://github.com/google/ml_collections/tree/master/ml_collections/config_flags/config_flags.py
  from ml_collections.config_flags.config_flags import _LoadConfigModule
  config_module = _LoadConfigModule("my_config_module", FLAGS['config'].config_filename)
  return config_module


def get_runname(cfg):
  from pathlib import Path
  parent_path = Path(__file__).parent
  model_name = parent_path.name

  config_module = load_config_module()
  runname = model_name + '-' + config_module.get_cfg_str(cfg)
  return runname


def get_config():
  cfg = FLAGS.config
  with cfg.unlocked():  # Save path to the config file; will later make a copy of it in exp dir.
    cfg.config_filename = FLAGS['config'].config_filename
  if FLAGS.hid is not None:  # Then we use the hid (work unit id) to index into a hparam config.
    config_module = load_config_module()
    hparam_cfg = config_module.get_hyper()[FLAGS.hid]
    logging.info("hid=%d, %s", FLAGS.hid, str(hparam_cfg))
    with cfg.unlocked():
      cfg.update_from_flattened_dict(hparam_cfg)
  return cfg


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  cfg = get_config()
  runname = get_runname(cfg)
  itinf_eval(cfg, model_cls=Model, experiments_dir=FLAGS.experiments_dir, runname=runname)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config'])
  app.run(main)
