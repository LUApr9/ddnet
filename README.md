# DD-Net: Dual decoder network with curriculum learning for full waveform inversion

## Abstract
Deep learning full waveform inversion (DL-FWI) is gaining much research interests due to its high prediction efficiency, effective exploitation of spatial correlation, and not requiring any initial estimate.  In this paper, we propose a dual decoder network with curriculum learning (DD-Net) to handle these issues. First, we design a U-Net with two decoders to grasp the velocity value and stratigraphic boundary information of the velocity model. Second, we introduce curriculum learning to model training by organizing data in three difficulty levels. Third, we generalize the model to new environments via a pre-network dimension reducer. 

![image](net/DDNet.png)

## >> Folder: (root directory)

### param_config.py
全局实验开关，决定数据集、输入输出尺寸、训练轮次、batch size、loss 权重等。
The global important variables of program operation are recorded, including some unique variables to each dataset.

### path_config
路径拼接中心；基于 dataset_name 动态映射到 data/*、models/*、results/*
The path where the program runs.
Among them, please modify the variable "main_dir" to your storage location.

### model_train.py (main running program)
主训练入口
The main program for training the our approaches.
The models generated during the training process will be stored in the models folder.
And the loss information generated in this process will be stored in results folder.

### inversionnet_train.py
The main program for training the InversionNet.
The models generated during the training process will be stored in the models folder.
And the loss information generated in this process will be stored in results folder.

### fcnvmb_train.py
The main program for training the FCNVMB.
The models generated during the training process will be stored in the models folder.
And the loss information generated in this process will be stored in results folder.

### model_test.py (main running program)
批量与单样本测试
The main program for testing the model.
The evaluation metric results generated during the test will be stored in the results folder.

## >> Folder: func
Store some commonly used function methods.

### datasets_reader.py
数据读取核心，支持 .mat（SEG）和 .npy（OpenFWI）
Several methods for reading seismic data and velocity models in batches and individually are documented.

### utils.py
轮廓提取（Canny）、可视化、指标、模型加载等工具。
Evaluation metrics and some common operations are documented.

## >> Folder: net
网络结构定义（DDNet、DDNet70、InversionNet、FCNVMB）
Some convolution operations and network architecture are documented.

### DDNet.py
Network architecture of DDNet and SDNet.

### DDNet70.py
Network architecture of DDNet70 and SDNet70.

### InversionNet.py
Network architecture of InversionNet.

### FCNVMB.py
Network architecture of FCNVMB.

## >> Folder: results
Store intermediate and final results of model runs.

## >> Folder: models
The path where the trained model is stored.
Each model is saved in the corresponding folder in .pkl format.

## >> Folder: data
Stores datasets documented in some papers.
The SEG dataset is stored one by one using .mat file, the OpenFWI dataset is stored using .npy files, and every 500 data is stored in one .npy file.
For specific dataset characteristics, see readme.md in each dataset folder.

## Quick Start: Pretrain -> Finetune -> Compare

This repo now supports a standalone source-domain pretraining flow.

1) Pretrain on source domains:

First edit `PRETRAIN_MANUAL_CONFIG` in `pretrain_entry.py`, then run:

```bash
python pretrain_entry.py
```

2) Finetune on target domain with pretrained checkpoint:

First edit `TRAIN_MANUAL_CONFIG` in `model_train.py`, then run:

```bash
python model_train.py
```

3) Compare baseline vs pretrained-finetuned models:

Set `TEST_MANUAL_CONFIG["compare_config"]` in `model_test.py`, then run:

```bash
python model_test.py
```

See `PRETRAIN_GUIDE.md` and `RUNBOOK.md` for detailed parameters and fallback behavior.
