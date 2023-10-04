This directory contains results on the Kodak, Tecnick, and CLIC professional validation datasets; see [paper](https://yiboyang.com/files/iccv23.pdf) for more details.

- `flops_per_pixel.csv` contains the flops/pixel for the (analysis, hyper analysis, synthesis, hyper synthesis) of various NTC methods in the paper.
You can load it like `import pandas as pd; fpp_df = pd.read_csv('results/flops_per_pixel.csv', index_col=0)`.

- `dataset/aggregate.json` contains the bits-per-pixel, PSNR, MS-SSIM, and LPIPS of the various methods on `dataset`, averaged over the images for each lambda.

- `dataset/{method}-detailed.json` contains the per-image R-D result for `method` trained with a particular `rd_lambda`, listed in no particular order. The `instance_id` gives the (0-based) index of the image being compressed (so `instance_id: 0` is `kodim01.png` for Kodak). There is one exception with `tecnick/2-layer_syn+SGA-detailed.json`: here SGA was run on batches of 5 images at a time to speed things up, so the `instance_id` is the batch id, and the corresponding R-D performance is the average of the batch.
