# -*- coding: utf-8 -*-


from param_config import *
import scipy.io
import scipy
import numpy as np
from func.utils import extract_contours

def batch_read_matfile(dataset_dir,
                       start,
                       batch_length,
                       train_or_test="train",
                       data_channels=29):
    '''
    批量读取 .mat 的地震数据与速度模型

    :param dataset_dir:             数据集路径
    :param start:                   起始读取编号
    :param batch_length:            从起始编号开始读取的长度
    :param train_or_test:           读取数据用于训练或测试（"train" 或 "test"）
    :param data_channels:           数据读取时使用的总通道数
    :return:                        a quadruple: (seismic data, [velocity model, contour of velocity model])
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity model are all (number of read data, channel, width x height)
    '''

    data_set = np.zeros([batch_length, data_channels, data_dim[0], data_dim[1]])
    label_set = np.zeros([batch_length, classes, model_dim[0], model_dim[1]])
    clabel_set = np.zeros([batch_length, classes, model_dim[0], model_dim[1]])

    for indx, i in enumerate(range(start, start + batch_length)):

        # 加载地震数据
        filename_seis = dataset_dir + '{}_data/seismic/seismic{}.mat'.format(train_or_test, i)
        print("Reading: {}".format(filename_seis))
        sei_data = scipy.io.loadmat(filename_seis)["data"]
        # (400, 301, 29) -> (29, 400, 301)
        sei_data = sei_data.swapaxes(0, 2)
        sei_data = sei_data.swapaxes(1, 2)
        for ch in range(inchannels):
            data_set[indx, ch, ...] = sei_data[ch, ...]

        # 加载速度模型
        filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.mat'.format(train_or_test, i)
        print("Reading: {}".format(filename_label))
        vm_data = scipy.io.loadmat(filename_label)["data"]
        label_set[indx, 0, ...] = vm_data
        clabel_set[indx, 0, ...] = extract_contours(vm_data)

    return data_set, [label_set, clabel_set]

def batch_read_npyfile(dataset_dir,
                       start,
                       batch_length,
                       train_or_test="train"):
    '''
    批量读取 .npy 的地震数据与速度模型

    :param dataset_dir:             数据集路径
    :param start:                   起始读取编号
    :param batch_length:            从起始编号开始读取的长度
    :param train_or_test:           读取数据用于训练或测试（"train" 或 "test"）
    :return:                        a pair: (seismic data, [velocity model, contour of velocity model])
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity
                                    model are all (number of read data * 500, channel, height, width)
    '''

    dataset = []
    labelset = []

    for i in range(start, start + batch_length):

        ##############################
        ##       加载地震数据       ##
        ##############################

        # 确定地震数据文件路径
        filename_seis = dataset_dir + '{}_data/seismic/seismic{}.npy'.format(train_or_test, i)
        print("Reading: {}".format(filename_seis))
        temp_data = np.load(filename_seis)

        dataset.append(temp_data)

        ##############################
        ##       加载速度模型       ##
        ##############################

        # 确定速度模型文件路径
        filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(train_or_test, i)
        print("Reading: {}".format(filename_label))
        temp_data = np.load(filename_label)

        labelset.append(temp_data)

    dataset = np.vstack(dataset)
    labelset = np.vstack(labelset)

    print("正在生成速度模型轮廓......")
    conlabels = np.zeros([batch_length * 500, classes, model_dim[0], model_dim[1]])
    for i in range(labelset.shape[0]):
        for j in range(labelset.shape[1]):
            conlabels[i, j, ...] = extract_contours(labelset[i, j, ...])

    return dataset, [labelset, conlabels]


def single_read_matfile(dataset_dir,
                        seismic_data_size,
                        velocity_model_size,
                        readID,
                        train_or_test = "train",
                        data_channels = 29):
    '''
    单样本读取 .mat 的地震数据与速度模型

    :param dataset_dir:             数据集路径
    :param seismic_data_size:       地震数据尺寸
    :param velocity_model_size:     速度模型尺寸
    :param readID:                  所选样本 ID
    :param train_or_test:           读取数据用于训练或测试（"train" 或 "test"）
    :param data_channels:           数据读取时使用的总通道数
    :return:                        a triplet: (seismic data, velocity model, contour of velocity model)
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity model are
                                    (channel, width, height), (width, height) and (width, height) respectively
    '''
    filename_seis = dataset_dir + '{}_data/seismic/seismic{}.mat'.format(train_or_test, readID)
    print("Reading: {}".format(filename_seis))
    filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.mat'.format(train_or_test, readID)
    print("Reading: {}".format(filename_label))

    se_data = scipy.io.loadmat(filename_seis)
    se_data = np.float32(se_data["data"].reshape([seismic_data_size[0], seismic_data_size[1], data_channels]))
    vm_data = scipy.io.loadmat(filename_label)
    vm_data = np.float32(vm_data["data"].reshape(velocity_model_size[0], velocity_model_size[1]))

    # (400, 301, 29) -> (29, 400, 301)
    se_data = se_data.swapaxes(0, 2)
    se_data = se_data.swapaxes(1, 2)

    contours_vm_data = extract_contours(vm_data)  # 使用 Canny 提取轮廓特征

    return se_data, vm_data, contours_vm_data

def single_read_npyfile(dataset_dir,
                        readIDs,
                        train_or_test = "train"):
    '''
    单样本读取 .npy 的地震数据与速度模型

    :param dataset_dir:             数据集路径
    :param readID:                  所选样本 ID 组合
    :param train_or_test:           读取数据用于训练或测试（"train" 或 "test"）
    :return:                        seismic data, velocity model, contour of velocity model
    '''

    # 确定地震数据文件路径
    filename_seis = dataset_dir + '{}_data/seismic/seismic{}.npy'.format(train_or_test, readIDs[0])
    print("Reading: {}".format(filename_seis))
    # 确定速度模型文件路径
    filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(train_or_test, readIDs[0])
    print("Reading: {}".format(filename_label))

    se_data = np.load(filename_seis)[readIDs[1]]
    vm_data = np.load(filename_label)[readIDs[1]][0]

    print("正在生成速度模型轮廓......")
    conlabel = extract_contours(vm_data)

    return se_data, vm_data, conlabel
