import numpy as np
import tensorflow as tf
import os
import sys
import json
from common.immutabledict import immutabledict

EMPTY_DICT = immutabledict()


def get_even_divisors(num):
  import math
  for j in range(math.ceil(math.sqrt(num)), 0, -1):
    div, mod = divmod(num, j)
    if mod == 0:
      return j, div  # j <= div


def load_json(path):
  with tf.io.gfile.GFile(path, 'r') as f:
    return json.load(f)


def dump_json(obj, path):
  with tf.io.gfile.GFile(path, 'w') as f:
    return json.dump(obj, f, indent=2)


def visualize_image_batch(images: tf.Tensor, crop_to_max_dim=None, ncol=None):
  """
  Given a [B, H, W, C] batch of images, return a single [H', W', C] image that displays (crops of)
  the given images on a grid. The output image shape (H', W') will be the two closest divisors of B
  so the resulting img is as close to a square as possible.
  :param images:
  :param crop_to_max_dim: will take crops to ensure the images being displayed are at most say
    256x256, to make the visualization look nice.
  :return:
  """

  if crop_to_max_dim and images.shape[1] > crop_to_max_dim and images.shape[2] > crop_to_max_dim:
    from common.image_utils import center_crop_image
    images = center_crop_image(images, crop_to_max_dim, crop_to_max_dim)

  batch_size = images.shape[0]
  if ncol is not None:
    nrow = int(batch_size / ncol)
    assert batch_size == nrow * ncol, f"Batch size {batch_size} does not evenly divide into ncol=" \
                                      f"{ncol}"
  else:
    ncol, nrow = get_even_divisors(batch_size)

  vis_image = tf.reshape(images, [nrow, ncol] + images.shape[1:])
  vis_image = tf.transpose(vis_image, [0, 2, 1, 3, 4])  # More easily done in np with swapaxes(1,2)
  vis_image = tf.reshape(vis_image, [nrow * images.shape[1], ncol * images.shape[2], -1])
  return vis_image


class ClassBuilder(dict):
  """
  Example:
    class A:
      def __init__(self, arg1, arg2):
        ...

  class_builder = ClassBuilder(ClassA=A)
  ClassBuilder('ClassA', arg1='x', arg2='y') -> A('x', 'y')
  """

  def build(self, class_name, **kwargs):
    cls = self[class_name]
    return cls(**kwargs)


try:
  from configs import args_abbr
except:
  args_abbr = {}


def config_dict_to_str(cfg, record_keys=None, skip_falsy=True, prefix=None, args_abbr=args_abbr,
                       primary_delimiter='-', secondary_delimiter='_'):
  """
  Given a dictionary of cmdline arguments, return a string that identifies the training run.
  :param cfg:
  :param record_keys: an iterable of strings corresponding to the keys to record. Default (None) is
  to record every (k,v) pair in the given dict.
  :param skip_falsy: whether to skip keys whose values evaluate to falsy (0, None, False, etc.)
  :param use_abbr: whether to use abbreviations for long key name
  :param primary_delimiter: the char to delimit different key-value paris
  :param secondary_delimiter: the delimiter within each key or value string (e.g., when the value is a list of numbers)
  :return:
  """
  kv_strs = []  # ['key1=val1', 'key2=val2', ...]
  if record_keys is None:  # Use all keys.
    record_keys = iter(cfg)
  for key in record_keys:
    val = cfg[key]
    if skip_falsy and not val:
      continue

    if isinstance(val, (list, tuple)):  # e.g., 'num_layers: [10, 8, 10] -> 'num_layers=10_8_10'
      val_str = secondary_delimiter.join(map(str, val))
    else:
      val_str = str(val)

    if args_abbr:
      key = args_abbr.get(key, key)

    kv_strs.append('%s=%s' % (key, val_str))

  if prefix:
    substrs = [prefix] + kv_strs
  else:
    substrs = kv_strs
  return primary_delimiter.join(substrs)


def get_xid():
  # See https://slurm.schedmd.com/job_array.html#env_vars
  xid = os.environ.get("SLURM_ARRAY_JOB_ID", None)
  if xid:
    return xid
  xid = os.environ.get("SLURM_JOB_ID", None)
  if xid:
    return xid
  return get_time_str()


def get_wid():
  return os.environ.get("SLURM_ARRAY_TASK_ID", None)


def log_run_info(workdir):
  run_info = {}
  run_info['cmdline'] = " ".join(
    sys.argv)  # attempt to reconstruct the original cmdline; not reliable (e.g., loses quotes)
  run_info['most_recent_version'] = get_git_revision_short_hash()

  for env_var in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID"):  # (xid, wid)
    if env_var in os.environ:
      run_info[env_var] = os.environ[env_var]

  import socket
  run_info['host_name'] = socket.gethostname()
  with open(os.path.join(workdir, f"run_info.json"), "w") as f:
    json.dump(run_info, f, indent=2)


# common_utils

def parse_runname(s, parse_numbers=False):
  """
  Given a string, infer key,value pairs that were used to generate the string and return the corresponding dict.
  Assume that the 'key' and 'value' are separated by '='. Care is taken to handle numbers in scientific notations.
  :param s: a string to be parsed.
  :param parse_numbers: if True, will try to convert values into floats (or ints) if possible; this may potentially
  lose information. By default (False), the values are simply strings as they appear in the original string.
  :return: an ordered dict, with key,value appearing in order
  >>> parse_runname('dir-lamb=2-arch=2_4_8/tau=1.0-step=0-kerasckpt')
  OrderedDict([('lamb', '2'), ('arch', '2_4_8')), ('tau', '1.0'), ('step', '0')])
  >>> parse_runname('rd-ms2020-latent_depth=320-hyperprior_depth=192-lmbda=1e-06-epoch=300-dataset=basenji-data_dim=4-bpp=0.000-psnr=19.875.npz')
  OrderedDict([('latent_depth', '320'),
               ('hyperprior_depth', '192'),
               ('lmbda', '1e-06'),
               ('epoch', '300'),
               ('dataset', 'basenji'),
               ('data_dim', '4'),
               ('bpp', '0.000'),
               ('psnr', '19.875')])
  """
  from collections import OrderedDict
  import re
  # Want to look for key, value pairs, of the form key_str=val_str.
  # In the following regex, key_str and val_str correspond to the first and second capturing groups, separated by '='.
  # The val_str should either correspond to a sequence of integers separated by underscores (like '2_3_12'), or a
  # numeric expression (possibly in scientific notation), or an alphanumeric string; the regex search is lazy and will
  # stop at the first possible match, in this order.
  # The sub-regex for scientific notation is adapted from https://stackoverflow.com/a/4479455
  sequence_delimiter = "_"
  pattern = fr'(\w+)=((\d+{sequence_delimiter})+\d+|(-?\d*\.?\d+(?:e[+-]?\d+)?)+|\w+)'

  def parse_ints(delimited_ints_str):
    ints = tuple(map(int, delimited_ints_str.split(sequence_delimiter)))
    return ints

  res = OrderedDict()
  for match in re.finditer(pattern, s):
    key = match.group(1)
    val = match.group(2)
    if match.group(3) is not None:  # Non-trivial match for a sequence of ints.
      if parse_numbers:
        val = parse_ints(val)
    else:  # Either matched a float-like number, or some string (\w+).
      if parse_numbers:
        try:
          val = float(val)
          if val == int(val):  # Parse to int if this can be done losslessly.
            val = int(val)
        except ValueError:
          pass
    res[key] = val
  return res


def preprocess_float_dict(d, format_str='.6g', as_str=False):
  # preprocess the floating values in a dict so that json.dump(dict) looks nice
  import tensorflow as tf
  import numpy as np
  res = {}
  for (k, v) in d.items():
    if isinstance(v, (float, np.floating)) or tf.is_tensor(v):
      if as_str:
        res[k] = format(float(v), format_str)
      else:
        res[k] = float(format(float(v), format_str))
    else:  # if not some kind of float, leave it be
      res[k] = v
  return res


def get_time_str(strftime_format="%Y,%m,%d,%H%M%S"):
  import datetime
  if not strftime_format:
    from configs import strftime_format

  time_str = datetime.datetime.now().strftime(strftime_format)
  return time_str


def psnr_to_float_mse(psnr):
  return 10 ** (-psnr / 10)


def float_mse_to_psnr(float_mse):
  return -10 * np.log10(float_mse)


# My custom logging code for logging in JSON lines ("jsonl") format
import json


class MyJSONEncoder(json.JSONEncoder):
  # https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return super(MyJSONEncoder, self).default(obj)


def get_json_logging_callback(log_file_path, buffering=1, **preprocess_float_kwargs):
  # Modified JSON logger example from https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LambdaCallback
  # Default is minimal buffering (=1)
  log_file = open(log_file_path, mode='wt', buffering=buffering)
  json_logging_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs_dict: log_file.write(
      json.dumps({'epoch': epoch, **preprocess_float_dict(logs_dict, **preprocess_float_kwargs)},
                 cls=MyJSONEncoder) + '\n'),
    on_train_end=lambda logs: log_file.close()
  )
  return json_logging_callback


# https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
import subprocess


def get_git_revision_hash() -> str:
  return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_git_revision_short_hash() -> str:
  return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
