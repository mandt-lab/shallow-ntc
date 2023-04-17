`flops_per_pixel.csv` contains the flops/pixel for the (analysis, hyper analysis, synthesis, hyper synthesis) of various NTC methods in the paper.
You can load it like `import pandas as pd; fpp_df = pd.read_csv('results/flops_per_pixel.csv', index_col=0)`.

`kodak/aggregate.json` contains the bpp, PSNR (RGB), MS-SSIM (RGB) of the proposed methods on Kodak, averaged over the images for each lambda.

`kodak/{method}-detailed.json` contains the per-image R-D result for `method` trained with a particular `rd_lambda`, listed in no particular order. The `instance_id` gives the (0-based) index of the image being compressed (so `instance_id: 0` is `kodim01.png` for Kodak).
