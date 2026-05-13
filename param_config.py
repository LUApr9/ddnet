# -*- coding: utf-8 -*-

####################################################
####                 主参数                    ####
####################################################

# 可选数据集: SEGSalt|SEGSimulation|FlatVelA|CurveFaultA|FlatFaultA|CurveVelA)
dataset_name = 'CurveFaultA'
learning_rate = 0.001                               # 学习率
classes = 1                                         # 输出通道数
display_step = 2                                    # 打印一次 loss 所需训练步数
model_type = 'DDNet70'
device_mode = 'cpu'                                # auto|cpu|gpu（训练/测试脚本通用）

####################################################
####               数据集参数                 ####
####################################################

####    如果数据集是.mat，size值为文件个数   ####
####    如果数据集是.npyt，size值为训练数据条数   ####



if dataset_name == 'FlatVelA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 1000
    test_size = 1000

    firststage_epochs = 10
    secondstage_epochs = 10
    thirdstage_epochs = 120
    loss_weight = [1, 0.01]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 64
    test_batch_size = 5

elif dataset_name == 'CurveVelA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 1000
    test_size = 1000

    firststage_epochs = 10
    secondstage_epochs = 10
    thirdstage_epochs = 120
    loss_weight = [1, 0.1]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 64
    test_batch_size = 5

elif dataset_name == 'FlatFaultA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 1000
    test_size = 1000

    firststage_epochs = 10
    secondstage_epochs = 10
    thirdstage_epochs = 120
    loss_weight = [1, 0.01]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 64
    test_batch_size = 5

elif dataset_name == 'CurveFaultA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    # train_size = 48000
    train_size = 1000
    # test_size = 6000
    test_size = 1000

    firststage_epochs = 10
    secondstage_epochs = 10
    thirdstage_epochs = 120
    loss_weight = [1, 0.1]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 64
    test_batch_size = 5

else:
    print('The selected dataset is invalid')
    exit(0)
