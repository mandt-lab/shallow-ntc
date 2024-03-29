# Executable for evaluating trained models.
# e.g., python eval.py --workdir train_xms/new/21965/mshyper-rd_lambda=0.08-latent_ch=320-base_ch=192 --dataset kodak

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
from common.data_lib import get_dataset
from common.eval_lib import eval_workdir
import tensorflow as tf
import os

FLAGS = flags.FLAGS

# config_flags.DEFINE_config_file('config', None, 'File path to the eval configuration.',
#                                 lock_config=True)
flags.DEFINE_string('workdir', None, "workdir to evaluate. This is generated by train_lib.")
flags.DEFINE_string('models_path', None, "Path to the models.py src defining the model class."
                                         "By default, use the copy from the workdir.")
flags.DEFINE_string('dataset', None, 'Dataset to eval.')
flags.DEFINE_integer('batchsize', 1, 'Size of eval data batches.')
flags.DEFINE_integer('patchsize', None, 'Size of cropped patches (default is no cropping).')
flags.DEFINE_boolean('profile', False, 'Whether to run in profile mode using tf.functions.')
flags.DEFINE_string('results_dir', None, 'Directory to store results.')
flags.DEFINE_boolean('skip_existing', True, 'Set to False to overwrite existing results files.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  eval_data = get_dataset(FLAGS.dataset, split="test", batchsize=FLAGS.batchsize,
                          patchsize=FLAGS.patchsize, normalize=True)

  results_dir = noprofile_results_dir = FLAGS.results_dir
  if results_dir is None:
    results_dir = noprofile_results_dir = f"./json_results/{FLAGS.dataset}/end_to_end"

  if FLAGS.profile:  # Will store results in a subdir, whose name is auto-generated based on hardware info.
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    device = 'gpu' if num_gpus else 'cpu'
    import socket
    host = socket.gethostname()
    results_dir = os.path.join(results_dir, "profile", f"device={device}-host={host}")

  if FLAGS.models_path is not None:
    import imp
    models = imp.load_source("my_models", FLAGS.models_path)
    model_cls = models.Model
  else:
    model_cls = None
  results_file_path = eval_workdir(FLAGS.workdir, eval_data, results_dir, model_cls=model_cls,
                                   profile=FLAGS.profile,
                                   skip_existing=FLAGS.skip_existing)

  if FLAGS.profile:
    # Make a symlink of the result in a common location (where results would be stored if eval
    # was run in non-profile mode), to simplify loading results in jupyter notebooks.
    from pathlib import Path
    noprofile_results_dir = Path(noprofile_results_dir)
    file_name = Path(results_file_path).name
    relative_results_dir = Path(os.path.join("profile", f"device={device}-host={host}"))
    symlink = Path(noprofile_results_dir / results_file_path.name)
    if symlink.exists() and not FLAGS.skip_existing:
      logging.info(f"Overwriting existing symlink: {symlink}")
      symlink.unlink()
    symlink.symlink_to(relative_results_dir / file_name)


if __name__ == '__main__':
  flags.mark_flags_as_required(['workdir', 'dataset'])
  app.run(main)
