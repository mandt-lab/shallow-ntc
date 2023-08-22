#!/usr/bin/env python
# Script for launching eval jobs on slurm. Typicaly each workdir corresponds to a single run /
# hyperparameter setting.
# Use like "./launch_eval.py --workdirs 'train_xms/new/*/*' --dataset kodak --sargs '-x ava-m5'"
from absl import app
from absl import flags
from absl import logging
from configs import project_dir, slurm_jobs_dir
from slurm_template import template
import subprocess

FLAGS = flags.FLAGS

flags.DEFINE_string('workdirs', None, 'Glob pattern for the workdirs to evaluate.')
flags.DEFINE_string('dataset', None, 'Glob pattern for the workdirs to evaluate.')
flags.DEFINE_string('args', "", 'Additional cmdline args for the eval module')
flags.DEFINE_string('sargs', '', 'Cmdline args for slurm; useful for custom resource specification')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.sargs:
    print(f"Using sargs = {FLAGS.sargs}")

  import glob
  workdirs = glob.glob(FLAGS.workdirs)  # list

  for i, workdir in enumerate(workdirs):
    print(f"\t{i}: {workdir}")

    # Set up template vars for slurm job script.
    job_name = f"eval_{FLAGS.dataset}:{i}"
    srun_command = f"srun python eval.py --workdir {workdir} --dataset {FLAGS.dataset} {FLAGS.args}"

    # Create slurm job.
    # Use a temp file name for now; will rename into job_id.job after submission.
    import uuid
    job_file_path = slurm_jobs_dir / str(uuid.uuid4())
    job_str = template.format(job_name=job_name, slurm_jobs_dir=slurm_jobs_dir,
                              last_hid=0, project_dir=project_dir,
                              srun_command=srun_command)
    with open(job_file_path, "w") as f:
      f.write(job_str)

    res = subprocess.check_output(f"sbatch {FLAGS.sargs} --mail-user= --parsable {job_file_path}",
                                  shell=True)  # Setting mail-user to empty disables email.
    job_id = int(res)
    job_file_path = job_file_path.rename(job_file_path.with_name(f"{job_id}.job"))
    print(f"Submitted job {job_id}; job script saved at {job_file_path}")


if __name__ == '__main__':
  flags.mark_flags_as_required(['workdirs', 'dataset'])
  app.run(main)
