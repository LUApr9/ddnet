# -*- coding: utf-8 -*-
"""
网络结构定义

Created on Feb 2023

@author: Xing-Yi Zhang (zxy20004182@163.com)

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################
#             常规网络单元                 #
#      （对应论文图1中的红色箭头）            #
###############################################

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, activ_fuc = nn.ReLU(inplace=True)):
        '''
        常规网络单元
        （对应论文图1中的红色箭头）

        :param in_size:      输入通道数
        :param out_size:     输出通道数
        :param is_batchnorm: 是否使用 BN
        :param activ_fuc:    激活函数
        '''
        super(unetConv2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       activ_fuc)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       activ_fuc)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       activ_fuc)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       activ_fuc)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

##################################################
#               下采样单元                     #
#      （对应论文图1中的紫色箭头）                #
##################################################

class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, activ_fuc = nn.ReLU(inplace=True)):
        '''
        下采样单元
        （对应论文图1中的紫色箭头）

        :param in_size:      输入通道数
        :param out_size:     输出通道数
        :param is_batchnorm: 是否使用 BN
        :param activ_fuc:    激活函数
        '''
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm, activ_fuc)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.down(outputs)
        return outputs

##################################################
#               上采样单元                     #
#      （对应论文图1中的黄色箭头）                #
##################################################

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, activ_fuc = nn.ReLU(inplace=True)):
        '''
        上采样单元
        （对应论文图1中的黄色箭头）

        :param in_size:      输入通道数
        :param out_size:     输出通道数
        :param is_deconv:    是否使用反卷积
        :param activ_fuc:    激活函数
        '''
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, True, activ_fuc)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset1 = (outputs2.size()[2] - inputs1.size()[2])
        offset2 = (outputs2.size()[3] - inputs1.size()[3])
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]

        # 跳跃连接并拼接
        outputs1 = F.pad(inputs1, padding)

        return self.conv(torch.cat([outputs1, outputs2], 1))

############################################
#              DD-Net 网络结构              #
############################################

class DDNetModel(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        '''
        DD-Net 网络结构

        :param n_classes:    输出通道数（单个解码器）
        :param in_channels:  网络输入通道数
        :param is_deconv:    是否使用反卷积
        :param is_batchnorm: 是否使用 BN
        '''
        super(DDNetModel, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        filters = [64, 128, 256, 512, 1024]

        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)

        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        
        self.dc1_up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.dc1_up3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.dc1_up2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.dc1_up1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.dc1_final = nn.Conv2d(filters[0], self.n_classes, 1)

        self.dc2_up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.dc2_up3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.dc2_up2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.dc2_up1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.dc2_final = nn.Conv2d(filters[0], 2, 1)

    def forward(self, inputs, label_dsp_dim):
        '''

        :param inputs:          输入图像
        :param label_dsp_dim:   网络输出图像尺寸（速度模型大小）
        :return:
        '''
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        center = self.center(down4)

        # 25*19*1024
        decoder1_image = center
        decoder2_image = center

        #################
        ###  解码器1 ###
        #################
        dc1_up4 = self.dc1_up4(down4, decoder1_image)
        dc1_up3 = self.dc1_up3(down3, dc1_up4)
        dc1_up2 = self.dc1_up2(down2, dc1_up3)
        dc1_up1 = self.dc1_up1(down1, dc1_up2)

        dc1_capture = dc1_up1[:, :, 1:1 + label_dsp_dim[0], 1:1 + label_dsp_dim[1]].contiguous()

        #################
        ###  解码器2 ###
        #################
        dc2_up4 = self.dc2_up4(down4, decoder2_image)
        dc2_up3 = self.dc2_up3(down3, dc2_up4)
        dc2_up2 = self.dc2_up2(down2, dc2_up3)
        dc2_up1 = self.dc2_up1(down1, dc2_up2)

        dc2_capture = dc2_up1[:, :, 1:1 + label_dsp_dim[0], 1:1 + label_dsp_dim[1]].contiguous()

        return [self.dc1_final(dc1_capture), self.dc2_final(dc2_capture)]

class LossDDNet:
    def __init__(self, weights=[1, 1], entropy_weight=[1, 1]):
        '''
        定义 DDNet 损失函数

        :param weights:         损失计算中两个解码器的权重。
        :param entropy_weight:  第二解码器两个输出通道的权重。
        '''

        self.criterion1 = nn.MSELoss()
        self.entropy_weight = torch.tensor(entropy_weight, dtype=torch.float32)
        self.weights = weights

    def __call__(self, outputs1, outputs2, targets1, targets2):
        '''

        :param outputs1: 第一解码器输出
        :param outputs2: 速度模型
        :param targets1: 第二解码器输出
        :param targets2: 速度模型轮廓
        :return:
        '''
        mse = self.criterion1(outputs1, targets1)
        cross = F.cross_entropy(
            outputs2,
            torch.squeeze(targets2).long(),
            weight=self.entropy_weight.to(outputs2.device)
        )

        criterion = (self.weights[0] * mse + self.weights[1] * cross)

        return criterion

############################################
#             SD-Net 网络结构               #
############################################

class SDNetModel(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_batchnorm):
        '''

        :param n_classes: 输出通道数（单个解码器）
        :param in_channels: 网络输入通道数
        :param is_deconv: 是否使用反卷积
        :param is_batchnorm: 是否使用 BN
        '''

        super(SDNetModel, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        filters = [64, 128, 256, 512, 1024]

        self.down1 = unetDown(self.in_channels, filters[0], self.is_batchnorm)


        self.down2 = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3 = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4 = unetDown(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.up4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up1 = unetUp(filters[1], filters[0], self.is_deconv)
        self.dc1_final = nn.Conv2d(filters[0], self.n_classes, 1)
        self.dc2_final = nn.Conv2d(filters[0], 2, 1)

    def forward(self, inputs, label_dsp_dim):
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        center = self.center(down4)

        dc1_up4 = self.up4(down4, center)
        dc1_up3 = self.up3(down3, dc1_up4)
        dc1_up2 = self.up2(down2, dc1_up3)
        dc1_up1 = self.up1(down1, dc1_up2)

        dc1_capture = dc1_up1[:, :, 1:1 + label_dsp_dim[0], 1:1 + label_dsp_dim[1]].contiguous()

        return self.dc1_final(dc1_capture)

if __name__ == '__main__':

    # 模型输出尺寸测试（DD-Net）

    x = torch.zeros((2, 29, 400, 301))
    model = DDNetModel(n_classes = 1,
                       in_channels = 29,
                       is_deconv = True,
                       is_batchnorm = True)
    out = model(x, label_dsp_dim = [201, 301])
    print("out1: {}, out2: {}".format(str(out[0].size()), str(out[1].size())))
