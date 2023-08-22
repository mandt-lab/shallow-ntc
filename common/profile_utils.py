"""
Miscellaneous utilities for profiling models.
"""
from typing import Optional, Union
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
  convert_variables_to_constants_v2_as_graph,
)

# import keras_flops.flops_registory

import functools
import time


def get_flops(model: Union[tf.keras.Model, tf.keras.Sequential], batch_size: Optional[int] =
None, stdout=False):
  """
  My mod of the get_flops function to allow turning off stdout logging and returning full flop
  stats.
  Original code: https://github.com/tokusumi/keras-flops/blob/master/keras_flops/flops_calculation.py
  Helpful references:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/python_api.md
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/options.m
  https://jiayiliu.github.io/posts/tensoflow-profiling/
  :param model:
  :param batch_size:
  :return:
  """
  if not isinstance(model, (tf.keras.Sequential, tf.keras.Model)):
    raise KeyError(
      "model arguments must be tf.keras.Model or tf.keras.Sequential instanse"
    )

  if batch_size is None:
    batch_size = 1

  # convert tf.keras model into frozen graph to count FLOPS about operations used at inference
  # FLOPS depends on batch size
  inputs = [
    tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs
  ]
  real_model = tf.function(model).get_concrete_function(inputs)
  frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

  # Calculate FLOPS with tf.profiler
  run_meta = tf.compat.v1.RunMetadata()
  opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
  # opts = tf.compat.v1.profiler.ProfileOptionBuilder.time_and_memory()  # Giving me zero execution time...seems impossible.
  # opts['min_micros'] = 0
  if not stdout:
    opts['output'] = 'none'
  flops = tf.compat.v1.profiler.profile(
    graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
  )
  # print(frozen_func.graph.get_operations())

  # return flops.total_float_ops
  return flops


def with_timing(func):
  """
  Decorator for timing a function. The decorated function will return a tuple of the original
  return value and the time taken to execute the function.
  :param func:
  :return:
  """
  @functools.wraps(func)
  def timeit_wrapper(*args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    return (result, total_time)

  return timeit_wrapper
