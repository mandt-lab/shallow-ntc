#!/usr/bin/env python
# Script for launching slurm jobs (usually for training).
from absl import app
from absl import flags
from absl import logging
import shutil
import os
from pathlib import Path
from configs import project_dir, slurm_jobs_dir
from slurm_template import template
import subprocess

FLAGS = flags.FLAGS

# Use like "./launch.py --main mshyper.train --config mshyper/configs/rd_lambda.py --args '--experiments_dir /tmp/test_slurm' --sargs '-w ava-m1'"
flags.DEFINE_string('main', None, 'Main module to run.')
flags.DEFINE_string('config', None, 'Path to the config file defining hparams.')
flags.DEFINE_string('args', "", 'Cmdline args for the main module')
flags.DEFINE_string('sargs', "", 'Cmdline args for slurm; useful for custom resource specification')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print(f"Using sargs = {FLAGS.sargs}")
  # Import config module, use get_hyper to get num hparam settings / num_workunits
  import imp
  config_module = imp.load_source("my_config_module", FLAGS.config)
  hparam_cfgs = config_module.get_hyper()
  print("Using the following hparams:")
  for i, cfg in enumerate(hparam_cfgs):
    print(f"\t{i}: {str(cfg)}")

  # Create a temp ID and copy the config file to a unique location. This is so that when the slurm
  # job is actually dispatched, the srun command will run with the version of config at the time of
  # job *submission*, rather than the current version of config file (which may have been modified
  # since submitted).
  import uuid
  my_id = str(uuid.uuid4())
  # Let's just save to slurm_jobs_dir for convenience.
  config_copy_path = slurm_jobs_dir / (my_id + '_' + os.path.basename(FLAGS.config))
  # like 'slurm_jobs/3dajdcow0e03_rd_lambda.py'
  shutil.copy2(FLAGS.config, config_copy_path)

  # Set up template vars for slurm job script.
  job_name = FLAGS.main
  last_hid = len(hparam_cfgs) - 1  # hid/wid will go from 0 to last_hid (inclusive)
  srun_command = f"srun python -m {FLAGS.main} --config {config_copy_path} --hid $SLURM_ARRAY_TASK_ID" \
                 f" {FLAGS.args}"
  # Note that $SLURM_ARRAY_TASK_ID will be set by slurm dynamically for each work unit.

  # Write sbatch job to file, to be submitted.
  # Use a temp file name for now; will rename into job_id.job after submission.
  job_file_path = slurm_jobs_dir / (my_id + '.job')
  job_str = template.format(job_name=job_name, slurm_jobs_dir=slurm_jobs_dir,
                            last_hid=last_hid, project_dir=project_dir,
                            srun_command=srun_command
                            )
  with open(job_file_path, "w") as f:
    f.write(job_str)

  # Create slurm array jobs with sbatch.
  res = subprocess.check_output(f"sbatch {FLAGS.sargs} --parsable {job_file_path}", shell=True)
  job_id = int(res)
  job_file_path = job_file_path.rename(job_file_path.with_name(f"{job_id}.job"))
  print(f"Submitted job {job_id}; job script saved at {job_file_path}")

  # Make a symlink to config_copy_path with Slurm assigned job id for easier future reference.
  Path(slurm_jobs_dir / f"{job_id}_config.py").symlink_to(Path(config_copy_path))


if __name__ == '__main__':
  flags.mark_flags_as_required(['main', 'config'])
  app.run(main)
