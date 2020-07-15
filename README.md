#  Att-McLSTM

This is an official PyTorch implementation of the Att-McLSTM model presented in the following paper:

>  Attention-based recurrent neural network for influenza epidemic prediction

## General

Project directories:

* `data`: data analysis and pre-processing
* `model`: source code
* `experiment`: comparative experiment and ablation study
* `utils`: some useful scripts for logging and showing demos

## Dependencies

This project uses Python 3.6 and PyTorch 0.4.1. First, create a conda environment and activate it: 

```
conda create -n att-mclstm python=3.6
conda activate att-mclstm
```

Note that this requires CUDA 9.2. Depending on your cuda version, you may want to install previous versions of PyTorch.  See [here](https://pytorch.org/get-started/previous-versions/).

