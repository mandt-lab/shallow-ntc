from pathlib import Path

project_dir = Path("/home/uname/projects/ntc")  # Set this to the root dir of this project.
slurm_jobs_dir = project_dir / "slurm_jobs"  # This is where Slurm jobs/logs are stored; can be anywhere.

strftime_format = "%Y,%m,%d,%H%M%S"

fixed_size_tfds_datasets = [  # Names of datasets that have fixed-size images.
  'mnist', 'cifar10', 'cifar100'
]

# Define your dataset name -> image glob mappings here.
dataset_to_globs = {
  'kodak': '/home/uname/data/kodak/kodim*.png',
  'kodak_landscape': '/home/uname/data/kodak/landscape/*.png',
  'tecnick': '/home/uname/data/Tecnick_TESTIMAGES/RGB/RGB_OR_1200x1200/*.png',
  'pval': '/home/uname/data/clic/pvalid/*.png',  # CLIC professional validation set
  'coco': '/home/uname/data/coco2017/*.png',
}

# Used for shortening the names of hyperparameters in runnames; optional.
args_abbr = {
  "channels_base": "base_ch",
  "bottleneck_size": "latent_ch"
}
