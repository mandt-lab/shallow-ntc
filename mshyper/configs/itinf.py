"""Hyperparameters for iterative inference experiments.
Modify the warm_start_exp_dir to point to the experiment directory and warm_start_wid to the work
unit dirs in it; I usually sweep over the latter to run SGA for each warm_start_wid in the
experiment dir to trace out an R-D curve.
"""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.data_config = dict(
    dataset="kodak_landscape",  # All the kodak images in landscape mode (512x768).
    batchsize=1,
    # Setting a batchsize larger than one can make the optimization finish faster, but requires
    # images to all have the same shape and the code will crash otherwise.
    patchsize=None,
  )
  config.train_eval_config = dict(
    num_steps=3000,
    log_metrics_every_steps=100,
    eval_every_steps=200,
    warm_start_exp_dir="project_dir/train_xms/my_xid",  # Set this to the experiment dir from
    # training (the xid is typically auto assigned by Slurm).
    warm_start_wid=0,  # Set this to the work unit id (wid) of the checkpoint to load.
  )

  config.model_config = dict(
    # The unspecified settings (e.g., rd_lambda) will be loaded from the checkpoint, so we don't
    # have to worry about setting them here (unless we want to override them).
    scheduled_num_steps=config.train_eval_config.num_steps,
    optimizer_config=dict(
      learning_rate=5e-3, reduce_lr_after=0.9, reduce_lr_factor=0.1,
      global_clipnorm=None, warmup_until=0.0
    ),
    latent_config=dict(
      uq=dict(
        # method="unoise",
        method="sga", tau_r=5e-4, tau_ub=0.5, tau_t0=200
      )
    ),
    offset_heuristic=False  # Should set this to False if training used mixedq
  )

  return config


def get_cfg_str(config):
  from collections import OrderedDict
  runname_dict = OrderedDict()
  runname_dict['wwid'] = config.train_eval_config.warm_start_wid
  runname_dict['uq_method'] = config.model_config.latent_config.uq.method

  from common import utils
  return utils.config_dict_to_str(runname_dict, skip_falsy=False)  # Avoid skipping 0 values.


def get_hyper():
  """
  Produce a list of flattened dicts, each containing a hparam configuration overriding the one in
  get_config(), corresponding to one hparam trial/experiment/work unit.
  :return:
  """
  from common import hyper
  wwids = list(range(7))  # Set this to the list of warmstart wids to run with.
  wwids = hyper.sweep('train_eval_config.warm_start_wid', wwids)
  hparam_cfgs = wwids
  return hparam_cfgs
