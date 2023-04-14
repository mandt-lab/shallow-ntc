`flops_per_pixel.csv` contains the flops/pixel for the (analysis, hyper analysis, synthesis, hyper synthesis) of various NTC methods in the paper.
You can load it like `import pandas as pd; fpp_df = pd.read_csv('results/flops_per_pixel.csv', index_col=0)`.

`kodak.json` contains the bpp, PSNR (RGB), MS-SSIM (RGB) of the proposed methods on Kodak.

