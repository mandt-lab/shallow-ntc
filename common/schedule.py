# coding=utf-8
# Adapted from https://github.com/google-research/google-research/blob/master/vct/src/schedule.py
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines functions for 'schedules', e.g. for a learning rate."""

import functools
import math
from typing import Sequence

import tensorflow as tf

TensorLike = tf.types.experimental.TensorLike
from enum import Enum


class InterpolationType(Enum):
  CONSTANT = "constant"
  LINEAR = "linear"
  SINE = "sine"


def piecewise_constant_schedule(step, boundaries, values):
  """Piecewise constant between boundaries and interval values."""
  # Also see https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PiecewiseConstantDecay
  # If no boundaries, function is constant.
  if len(values) != len(boundaries) + 1:
    raise ValueError("The number of values must be one more than the number "
                     f"of boundaries: {len(values)} != {len(boundaries) + 1}")
  step = tf.convert_to_tensor(step)
  # Cast `boundaries` to have the same type as `step`.
  boundaries = tf.convert_to_tensor(boundaries, dtype=step.dtype)
  values = tf.convert_to_tensor(values)
  index = tf.math.reduce_sum(
    tf.cast(boundaries <= tf.expand_dims(step, axis=-1), tf.int32), axis=-1)
  return tf.gather(values, index)


def piecewise_sine_schedule(step, boundaries, values):
  """My piecewise sine schedule.
   boundaries, values are interpreted as giving a list of (x,y) points (boundaries[0], values[0]),
   (boundaries[1], values[1]), ... (boundaries[-1], values[-1]). If step < boundaries[0] (leftmost
   boundary), then values[0] is returned; if step >=boundaries[-1] (rightmost boundary), then
   values[-1] is returned. For all step values in between, the returned result is obtained by
   interpolating the two neighboring (x,y) pairs.
  """

  if len(values) != len(boundaries):
    raise ValueError("The number of values must equal the number "
                     f"of boundaries: {len(values)} != {len(boundaries)}")

  step = tf.convert_to_tensor(step)
  # Cast `boundaries` to have the same type as `step`.
  boundaries = tf.convert_to_tensor(boundaries, dtype=step.dtype)
  values = tf.convert_to_tensor(values)
  comp = tf.expand_dims(step, axis=-1) >= boundaries
  right_end = tf.reduce_all(comp)
  left_end = tf.reduce_all(tf.math.logical_not(comp))

  left_val = lambda: values[0]
  right_val = lambda: values[-1]

  def interp_val():
    index = tf.math.reduce_sum(tf.cast(comp, tf.int32))
    x = tf.cast(step, dtype=values.dtype)
    boundaries_float = tf.cast(boundaries, dtype=values.dtype)
    xleft, xright = boundaries_float[index - 1], boundaries_float[index]
    yleft, yright = values[index - 1], values[index]

    pi = tf.constant(math.pi, dtype=values.dtype)
    interp = yleft + (yright - yleft) * tf.math.sin((x - xleft) / (xright - xleft) * 0.5 * pi)
    return interp

  pred_fn_pairs = [
    (left_end, left_val),
    (right_end, right_val)]

  return tf.case(pred_fn_pairs, default=interp_val)


def schedule_at_step(step,
                     vals,
                     boundaries,
                     interpolation,
                     warmup_steps=0
                     ):
  """Computes the schedule value at a given step `step`.
  Args:
    step: The step to compute the schedule value at.
    vals: Sequence of values.
    boundaries: Locations where the schedule changes between values in `vals`.
      If empty, `vals` should be a sequence with exactly one element.
    interpolation: Interpolation type to use.
    warmup_steps:  Number of steps at the beginning of training to use as
      warmup. Set to non-positive to disable.
  Returns:
    The computed schedule value at step `step`.
  """
  step = tf.convert_to_tensor(step)
  if len(boundaries) == 0:
    return tf.cast(tf.squeeze(vals), dtype='float32')
  if interpolation == InterpolationType.CONSTANT:
    value = piecewise_constant_schedule(step, boundaries, vals)
  elif interpolation == InterpolationType.SINE:
    value = piecewise_sine_schedule(step, boundaries, vals)
  else:
    raise NotImplementedError

  if warmup_steps > 0:
    # Applies linear warmup, over the first `warmup_steps` steps.
    value *= tf.minimum(1., (tf.cast(step, tf.float32) + 1) / warmup_steps)

  return value


class KerasSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Returns `schedule_at_step` above in the form of a KerasSchedule.
  Here the schedule is multiplicative over a provided base value.
  Example usage:
  learning_rate_schedule = schedule.KerasSchedule(
      base_value=0.1,
      vals=[8, 4, 2],
      boundaries=[10, 15],
      interpolation="linear",
  )
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
  """

  def __init__(self, base_value=1.0, **kwargs):
    """Initializes the schedule.
    Args:
      base_value: A base value that is multiplied with the scheduled value.
      **kwargs: Schedule configuration compatible with
        `schedules.schedule_at_step`.
    """
    self._base_value = tf.convert_to_tensor(base_value, tf.float32)
    self._schedule_at_step = functools.partial(schedule_at_step, **kwargs)

  def __call__(self, step):
    return self._base_value * self._schedule_at_step(step)


class CompressionSchedule(KerasSchedule):
  """LR Schedule for compression, with a drop at the end and warmup."""

  def __init__(
        self,
        base_learning_rate,
        total_num_steps,
        warmup_until=0,  # Keeping this arg for backwards compatibility.
        warmup_steps=None,  # If provided, will take precedence over the 'warmup_until' arg.
        drop_after=0.85,
        drop_factor=0.1,
  ):
    if warmup_steps is None:
      assert warmup_until is not None
      warmup_steps = int(warmup_until * total_num_steps)
    super().__init__(
      base_value=base_learning_rate,
      warmup_steps=warmup_steps,
      vals=[1., drop_factor],
      boundaries=[int(drop_after * total_num_steps)],
      interpolation=InterpolationType.CONSTANT
    )
