# -*- coding: utf-8 -*-


from param_config import *                                                      # 读取数据集名称
import os

###################################################
####                 路径                      #####
###################################################

main_dir        = 'D:/bishe/ddnet-master/'
data_dir        = main_dir + 'data/'                                            # 数据集路径
results_dir     = main_dir + 'results/'                                         # 运行结果输出路径（不含模型文件）
models_dir      = main_dir + 'models/'                                          # 训练完成后模型保存路径

###################################################
####               动态路径                    #####
###################################################

temp_results_dir= results_dir + '{}Results/'.format(dataset_name)               # 为指定数据集生成结果保存路径
temp_models_dir = models_dir + '{}Model/'.format(dataset_name)                 # 为指定数据集生成模型保存路径
data_dir = data_dir + '{}/'.format(dataset_name)                      # 为指定数据集生成数据路径

if os.path.exists(temp_results_dir) and os.path.exists(temp_models_dir):
    results_dir = temp_results_dir
    models_dir  = temp_models_dir
else:
    os.makedirs(temp_results_dir)
    os.makedirs(temp_models_dir)
    results_dir = temp_results_dir
    models_dir  = temp_models_dir
