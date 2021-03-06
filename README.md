#  Att-McLSTM

This is an official PyTorch implementation of the Att-McLSTM model presented in the following paper:

>  [Attention-based recurrent neural network for influenza epidemic prediction](https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/s12859-019-3131-8)

## General

Project directories:

* `data`: data analysis and pre-processing
* `model`: source code
* `experiment`: comparative experiments and ablation study
* `utils`: some useful scripts for logging and showing demos

## Dependencies

This project uses Python 3.6 and Keras. I recommend using anaconda for dependency management: 

```
conda create -n att-mclstm python=3.6
conda activate att-mclstm
```

Note that this requires CUDA 9.2. Depending on your cuda version, you may want to install previous versions of TensorFlow.  See [here](https://www.tensorflow.org/versions).

## Reproducibility

The results of all functions and training loop match the results demonstrated in the paper. One can check it out and run to test:

```
python ./data/sum_data.py
python ./model/att_multi_channel_lstm.py
```

## Misc

### Citation

If you find our work useful in your research, please cite:

```
@article{zhu2019attention,
  title={Attention-based recurrent neural network for influenza epidemic prediction},
  author={Zhu, Xianglei and Fu, Bofeng and Yang, Yaodong and Ma, Yu and Hao, Jianye and Chen, Siqi and Liu, Shuang and Li, Tiegang and Liu, Sen and Guo, Weiming and others},
  journal={BMC bioinformatics},
  volume={20},
  number={18},
  pages={1--10},
  year={2019},
  publisher={BioMed Central}
}
```

### Contact

If you have any questions, please email Bofeng Fu at bofeng_fu@163.com.

