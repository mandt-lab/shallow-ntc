"""A small extension of metric_writers' default writers to allow logging to json lines.
Based on https://github.com/google/CommonLoopUtils/blob/95994cbf2f05f477e8a72ec47b8d0b48549d1684/clu/metric_writers/logging_writer.py#L27
and
https://github.com/google/CommonLoopUtils/blob/95994cbf2f05f477e8a72ec47b8d0b48549d1684/clu/metric_writers/utils.py#L100

Yibo Yang, 2022.
"""

import os
from typing import Any, Mapping, Optional, Tuple
from clu.metric_writers.async_writer import AsyncMultiWriter
from clu.metric_writers.interface import MetricWriter
from clu.metric_writers.logging_writer import LoggingWriter
from clu.metric_writers.multi_writer import MultiWriter
from clu.metric_writers.summary_writer import SummaryWriter
from etils import epath
import numpy as np

from clu.metric_writers import interface
from pathlib import Path
import json
from collections import OrderedDict
from common.utils import preprocess_float_dict

Array = interface.Array
Scalar = interface.Scalar


class JsonlWriter(interface.MetricWriter):
  """MetricWriter that writes all values to json lines.
  Based on LoggingWriter https://github.com/google/CommonLoopUtils/blob/95994cbf2f05f477e8a72ec47b8d0b48549d1684/clu/metric_writers/logging_writer.py#L27
  """

  def __init__(self, logdir, file_name="record.jsonl", buffering=1):
    log_file_path = Path(logdir) / file_name
    self.file_handle = open(log_file_path, mode='w', buffering=buffering)

  def write_summaries(
        self, step: int,
        values: Mapping[str, Array],
        metadata: Optional[Mapping[str, Any]] = None):
    pass

  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
    # Skip logging only steps_per_sec from periodic_actions.ReportProgress.
    if len(scalars) == 1 and 'steps_per_sec' in scalars:
      return

    log_dict = OrderedDict(step=step)
    log_dict.update(scalars)
    log_dict = preprocess_float_dict(log_dict, format_str='.6f')
    json.dump(log_dict, self.file_handle)
    self.file_handle.write('\n')

  def write_images(self, step: int, images: Mapping[str, Array]):
    pass

  def write_videos(self, step: int, videos: Mapping[str, Array]):
    pass

  def write_audios(
        self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
    pass

  def write_texts(self, step: int, texts: Mapping[str, str]):
    log_dict = OrderedDict(step=step)
    log_dict.update(texts)
    json.dump(log_dict, self.file_handle)
    self.file_handle.write('\n')

  def write_histograms(self,
                       step: int,
                       arrays: Mapping[str, Array],
                       num_buckets: Optional[Mapping[str, int]] = None):
    pass

  def write_hparams(self, hparams: Mapping[str, Any]):
    pass

  def flush(self):
    self.file_handle.flush()

  def close(self):
    self.file_handle.close()


# Based on https://github.com/google/CommonLoopUtils/blob/95994cbf2f05f477e8a72ec47b8d0b48549d1684/clu/metric_writers/utils.py#L100
# The only difference is we also add our JsonlWriter to the MultiWriter.
def create_default_writer(
      logdir: Optional[epath.PathLike] = None,
      *,
      just_logging: bool = False,
      asynchronous: bool = True,
      collection: Optional[str] = None) -> MultiWriter:
  """Create the default writer for the platform.
  On most platforms this will create a MultiWriter that writes to multiple back
  ends (logging, TF summaries etc.).
  Args:
    logdir: Logging dir to use for TF summary files. If empty/None will the
      returned writer will not write TF summary files.
    just_logging: If True only use a LoggingWriter. This is useful in multi-host
      setups when only the first host should write metrics and all other hosts
      should only write to their own logs.
    write_to_xm_measurements: If True uses XmMeasurementsWriter in addition.
      default (None) will automatically determine if you # GOOGLE-INTERNAL have
    asynchronous: If True return an AsyncMultiWriter to not block when writing
      metrics.
    collection: A string which, if provided, provides an indication that the
      provided metrics should all be written to the same collection, or
      grouping.
  Returns:
    A `MetricWriter` according to the platform and arguments.
  """
  if just_logging:
    if asynchronous:
      return AsyncMultiWriter([LoggingWriter(collection=collection)])
    else:
      return MultiWriter([LoggingWriter(collection=collection)])
  writers = [LoggingWriter(collection=collection)]
  if logdir is not None:
    logdir = epath.Path(logdir)
    if collection is not None:
      logdir /= collection
    writers.append(SummaryWriter(os.fspath(logdir)))
    writers.append(JsonlWriter(os.fspath(logdir)))
  if asynchronous:
    return AsyncMultiWriter(writers)
  return MultiWriter(writers)
