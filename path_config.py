# -*- coding: utf-8 -*-
import os
import sys
from param_config import *

###################################################
####               动态根路径获取                #####
###################################################

# 如果是 PyInstaller 打包后的 exe，sys.executable 指向 exe 路径
# 如果是普通 python 运行，__file__ 指向当前脚本路径
if hasattr(sys, '_MEIPASS'):
    # 打包后的环境，BASE_DIR 为 exe 所在目录
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # 源码运行环境，BASE_DIR 为 path_config.py 所在目录（根目录）
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 统一使用绝对路径，避免由于启动位置不同导致的找不到文件
main_dir = BASE_DIR.replace('\\', '/') + '/'

###################################################
####                 基础路径                  #####
###################################################

data_root_dir   = os.path.join(main_dir, 'data/')
results_root_dir= os.path.join(main_dir, 'results/')
models_root_dir = os.path.join(main_dir, 'models/')

###################################################
####               动态子路径生成              #####
###################################################

# 根据数据集名称生成对应的子目录
results_dir = os.path.join(results_root_dir, '{}Results/'.format(dataset_name))
models_dir  = os.path.join(models_root_dir, '{}Model/'.format(dataset_name))
data_dir    = os.path.join(data_root_dir, '{}/'.format(dataset_name))

# 自动创建不存在的目录
for p in [results_dir, models_dir]:
    if not os.path.exists(p):
        os.makedirs(p)

print(f"[*] 路径初始化完成: {main_dir}")