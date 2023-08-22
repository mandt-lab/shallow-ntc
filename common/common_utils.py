# Commonly used utility routines for organizing/keeping track of my experiments.

def get_runname(args_dict, record_keys=tuple(), prefix=''):
  """
  Given a dictionary of cmdline arguments, return a string that identifies the training run.
  :param args_dict:
  :param record_keys: a tuple/list of keys that is a subset of keys in args_dict that will be used to form the runname
  :return:
  """
  kv_strs = []  # ['key1=val1', 'key2=val2', ...]

  for key in record_keys:
    val = args_dict[key]
    if isinstance(val, (list, tuple)):  # e.g., 'num_layers: [10, 8, 10] -> 'num_layers=10_8_10'
      val_str = '_'.join(map(str, val))
    else:
      val_str = str(val)
    kv_strs.append('%s=%s' % (key, val_str))

  return '-'.join([prefix] + kv_strs)


class AttrDict(dict):
  # https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self


def get_args_as_obj(args):
  """
  Get an object specifying options/hyper-params from a JSON file or a Python dict; simulates the result of argparse.
  No processing is done if the input is of neither type (assumed to already be in an obj format).
  :param args: either a dict-like object with attributes specifying the model, or the path to some args.json file
  containing the args (which will be loaded and converted to a dict).
  :return:
  """
  if isinstance(args, str):
    import json
    with open(args) as f:
      args = json.load(f)
  if isinstance(args, dict):
    args = AttrDict(args)
  return args


def config_dict_to_str(args_dict, record_keys=tuple(), leave_out_falsy=True, prefix=None,
                       use_abbr=False,
                       primary_delimiter='-', secondary_delimiter='_'):
  """
  Given a dictionary of cmdline arguments, return a string that identifies the training run.
  :param args_dict:
  :param record_keys: a tuple/list of keys that is a subset of keys in args_dict that will be used to form the runname
  :param leave_out_falsy: whether to skip keys whose values evaluate to falsy (0, None, False, etc.)
  :param use_abbr: whether to use abbreviations for long key name
  :param primary_delimiter: the char to delimit different key-value paris
  :param secondary_delimiter: the delimiter within each key or value string (e.g., when the value is a list of numbers)
  :return:
  """
  kv_strs = []  # ['key1=val1', 'key2=val2', ...]

  for key in record_keys:
    val = args_dict[key]
    if leave_out_falsy and not val:
      continue
    if isinstance(val, (list, tuple)):  # e.g., 'num_layers: [10, 8, 10] -> 'num_layers=10_8_10'
      val_str = secondary_delimiter.join(map(str, val))
    else:
      val_str = str(val)
    if use_abbr:
      from configs import cmdline_arg_abbr
      key = cmdline_arg_abbr.get(key, key)
    kv_strs.append('%s=%s' % (key, val_str))

  if prefix:
    substrs = [prefix] + kv_strs
  else:
    substrs = kv_strs
  return primary_delimiter.join(substrs)


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


def get_time_str():
  import datetime
  try:
    from configs import strftime_format
  except ImportError:
    strftime_format = "%Y_%m_%d~%H_%M_%S"

  time_str = datetime.datetime.now().strftime(strftime_format)
  return time_str


def natural_sort(l):
  # https://stackoverflow.com/a/4836734
  import re
  convert = lambda text: int(text) if text.isdigit() else text.lower()
  alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
  return sorted(l, key=alphanum_key)
