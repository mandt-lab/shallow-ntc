# Basic helper methods for producing hyperparameter configurations in the form of a list of dicts.

import itertools
from functools import reduce


def sweep(key, vals):
  return [{key: val} for val in vals]


def izip(*hparam_dicts):
  """
  Example:
   A = [{'a': 0}, {'a': 1}, {'a': 2}]
   B = [{'b': -2}, {'b': -1}]
   zip(A, B) gives
    [{'a': 0, 'b': -2},
     {'a': 1, 'b': -1}]
  :param hparam_dicts:
  :return:
  """
  hparam_dicts_zip = zip(*hparam_dicts)

  return [reduce(lambda d1, d2: {**d1, **d2}, tuple_of_dicts) for tuple_of_dicts in
          hparam_dicts_zip]


def product(*hparam_dicts):
  """
  Example:
   A = [{'a': 0}, {'a': 1}, {'a': 2}]
   B = [{'b': -2}, {'b': -1}]
   product(A, B) gives
    [{'a': 0, 'b': -2},
     {'a': 0, 'b': -1},
     {'a': 1, 'b': -2},
     {'a': 1, 'b': -1},
     {'a': 2, 'b': -2},
     {'a': 2, 'b': -1}]
  :param hparam_dicts:
  :return:
  """
  hparam_dicts_product = itertools.product(*hparam_dicts)

  return [reduce(lambda d1, d2: {**d1, **d2}, tuple_of_dicts) for tuple_of_dicts in
          hparam_dicts_product]
