# About

This repo contains the source code and results in the paper
[Asymmetrically-powered Neural Image Compression with Shallow Decoders](https://arxiv.org/abs/2304.06244).
As far as we know, this is the first neural image compression method to operate with a decoding budget of less than 50K FLOPs/pixel while achieving rate-distortion performance competitive with BPG.



# Software

The code was developed in a python 3.10 environment on Linux.
The main dependices are tensorflow 2.10 and [tensorflow-compression 2.10](https://github.com/tensorflow/compression/releases/tag/v2.10.0), which can be installed by
`pip install tensorflow-compression==2.10.0`




