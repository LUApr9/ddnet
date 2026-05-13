# -*- coding: utf-8 -*-

from path_config import *
from func.utils import model_reader, add_gasuss_noise, magnify_amplitude_fornumpy
from func.datasets_reader import batch_read_matfile, batch_read_npyfile
from net.InversionNet import InversionNet
from net.FCNVMB import FCNVMB
from net.DDNet70 import DDNet70Model, SDNet70Model, LossDDNet
from net.DDNet import DDNetModel, SDNetModel
from func.device_selector import get_runtime_device
from math import ceil
import os
import json

def _training_mode_dir():
    '''
    Determine training mode directory based on load_pretrained presence (implicit mode).
    Returns:
        'used_pretrain'  -> finetune from pretrained weights
        'unused_pretrain' -> from-scratch training
    '''
    load_path = TRAIN_MANUAL_CONFIG.get("load_pretrained", "")
    return "used_pretrain" if (load_path is not None and str(load_path).strip() != "") else "unused_pretrain"

import time
import numpy as np
import torch
import torch.utils.data as data_utils
import gc
import torch.nn.functional as F


def determine_network(external_model_src="", model_type="DDNet", lr=None):
    '''
    获取网络对象并导入外部模型，或创建初始化模型

    :param external_model_src:  外部 pkl 文件路径
    :param model_type:          主模型类型，根据不同论文实现进行区分。
                                可用模型关键字为
                                [DDNet | DDNet70 | InversionNet | FCNVMB | SDNet | SDNet70]
    :return:                    三元组：模型对象、设备对象与优化器
    '''

    device, cuda_available, resolved_mode = get_runtime_device(device_mode)
    print(
        "[Device] mode={} resolved={} cuda_available={}".format(resolved_mode, device.type, torch.cuda.is_available()))
    gpus = [0]

    # 网络初始化
    if model_type == "DDNet":
        net_model = DDNetModel(n_classes=classes,
                               in_channels=inchannels,
                               is_deconv=True,
                               is_batchnorm=True)
    elif model_type == "DDNet70":
        net_model = DDNet70Model(n_classes=classes,
                                 in_channels=inchannels,
                                 is_deconv=True,
                                 is_batchnorm=True)
    elif model_type == "SDNet":
        net_model = SDNetModel(n_classes=classes,
                               in_channels=inchannels,
                               is_deconv=True,
                               is_batchnorm=True)
    elif model_type == "SDNet70":
        net_model = SDNet70Model(n_classes=classes,
                                 in_channels=inchannels,
                                 is_deconv=True,
                                 is_batchnorm=True)
    elif model_type == "InversionNet":
        net_model = InversionNet()
    elif model_type == "FCNVMB":
        net_model = FCNVMB(n_classes=classes,
                           in_channels=inchannels,
                           is_deconv=True,
                           is_batchnorm=True)
    else:
        net_model = None
        print(
            'The "model_type" parameter selected in the determine_network(...)'
            ' is the undefined network model keyword! Please check!')
        exit(0)

    # 继承已有网络参数
    if external_model_src != "":
        net_model = model_reader(net=net_model, device=device, save_src=external_model_src)

    # 分配设备并设置优化器
    if cuda_available:
        # 旧版本写法：net_model = torch.nn.DataParallel(net_model.cuda(), device_ids=gpus)
        net_model = torch.nn.DataParallel(net_model.cuda(), device_ids=gpus)
    else:
        net_model = net_model.to(device)

    if lr is None:
        lr = learning_rate
    optimizer = torch.optim.Adam(net_model.parameters(), lr=lr)

    return net_model, device, optimizer


def load_dataset(stage=3):
    '''
    根据 "param_config" 中参数加载训练数据

    :return:
    '''

    print("---------------------------------")
    print("· Loading the datasets...")

    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        data_set, label_sets = batch_read_matfile(data_dir, 1, train_size, "train")
    else:
        data_set, label_sets = batch_read_npyfile(data_dir, 1, ceil(train_size / 500), "train")
        # data_set, label_sets = batch_read_npyfile(data_dir, 1, 3, "train")
        for i in range(data_set.shape[0]):
            vm = label_sets[0][i][0]
            label_sets[0][i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        middle_shot_id = 15
        first_p = 9
        second_p = 18
    else:
        middle_shot_id = 2
        first_p = 2
        second_p = 4

    if stage == 1:
        for eachData in range(train_size):
            middle_shot = data_set[eachData, middle_shot_id, :, :].copy()
            middle_shot_with_noise = add_gasuss_noise(middle_shot.copy())
            middle_shot_magnified = magnify_amplitude_fornumpy(middle_shot.copy())
            for j in range(second_p, inchannels):
                data_set[eachData, j, :, :] = middle_shot
            for j in range(first_p, second_p):
                data_set[eachData, j, :, :] = middle_shot_magnified
            for j in range(0, first_p):
                data_set[eachData, j, :, :] = middle_shot_with_noise
    elif stage == 2:
        for eachBatch in range(train_size):
            middle_shot = data_set[eachBatch, middle_shot_id, :, :].copy()
            for eachChannel in range(inchannels):
                data_set[eachBatch, eachChannel, :, :] = middle_shot
    else:
        pass

    # 训练集
    seis_and_vm = data_utils.TensorDataset(
        torch.from_numpy(data_set).float(),
        torch.from_numpy(label_sets[0]).float(),
        torch.from_numpy(label_sets[1]).long())
    seis_and_vm_loader = data_utils.DataLoader(
        seis_and_vm,
        batch_size=train_batch_size,
        pin_memory=True,
        shuffle=True)

    print("· Number of seismic gathers included in the training set: {}".format(train_size))
    print("· Dimensions of seismic data: ({},{},{},{})".format(train_size, inchannels, data_dim[0], data_dim[1]))
    print("· Dimensions of velocity model: ({},{},{},{})".format(train_size, classes, model_dim[0], model_dim[1]))
    print("---------------------------------")

    return seis_and_vm_loader, data_set, label_sets


def train_for_one_stage(cur_epochs, model, training_loader, optimizer, save_times=1, key_word="CLstage1",
                        model_type="DDNet"):
    '''
    按指定轮次进行训练

    :param cur_epochs:      指定训练轮数
    :param model:           用于训练的网络模型对象
    :param training_loader: 送入网络的训练集 DataLoader
    :param optimizer:       优化器
    :param key_word:        训练后用于模型命名的关键字
    :param stage_keyword:   所选难度阶段关键字（设为 no settings 可忽略 CL）
    :param model_type:      主模型类型，根据不同论文实现进行区分。
                            可用模型关键字为 [DDNet | DDNet70 | InversionNet | FCNVMB]
    :return:                模型保存路径
    '''

    # Determine output directory (implicit mode: used_pretrain/unused_pretrain)
    mode_dir = _training_mode_dir()
    per_run_root = os.path.join(results_dir, mode_dir, model_type, dataset_name)
    os.makedirs(per_run_root, exist_ok=True)

    loss_of_stage = []
    last_model_save_path = ""
    step = int(train_size / train_batch_size)  # 训练总 batch 数
    save_epoch = cur_epochs // save_times
    training_time = 0

    model_save_name = "{}_{}_{}_TrSize{}_AllEpo{}".format(dataset_name, key_word, mode_dir, train_size, cur_epochs)

    model_device = next(model.parameters()).device

    for epoch in range(cur_epochs):
        # 当前 epoch 的训练
        loss_of_epoch = 0.0
        batch_count = 0
        cur_node_time = time.time()
        ############
        # 训练
        ############
        for i, (images, labels, contours_labels) in enumerate(training_loader):
            batch_count += 1

            iteration = epoch * step + i + 1
            model.train()

            # 加载到设备
            # 旧版本写法：
            # if torch.cuda.is_available():
            #     images = images.cuda(non_blocking=True)
            #     labels = labels.cuda(non_blocking=True)
            #     contours_labels = contours_labels.cuda(non_blocking=True)
            images = images.to(model_device, non_blocking=model_device.type == "cuda")
            labels = labels.to(model_device, non_blocking=model_device.type == "cuda")
            contours_labels = contours_labels.to(model_device, non_blocking=model_device.type == "cuda")

            # 清空梯度缓存
            optimizer.zero_grad()
            criterion = LossDDNet(weights=loss_weight)

            #if model_type in ["DDNet", "DDNet70"]:
               # outputs = model(images, model_dim)
                #loss = criterion(outputs[0], outputs[1], labels, contours_labels)
            if model_type in ["DDNet", "DDNet70"]:
                outputs = model(images, model_dim)
                # 计算原始多任务损失
                loss = criterion(outputs[0], outputs[1], labels, contours_labels)

                # --- 新增纠偏逻辑开始 ---
                # 针对归一化后 0.4 到 0.6 的绿色中速层增加 5 倍惩罚
                # 这会强迫模型把被挤压的绿色层“画”出来
                green_mask = (labels > 0.4) & (labels < 0.6)
                if green_mask.any():
                    # 仅针对速度模型 outputs[0] 的绿色区域计算额外 MSE
                    extra_green_loss = F.mse_loss(outputs[0][green_mask], labels[green_mask]) * 5.0
                    loss = loss + extra_green_loss
                # --- 新增纠偏逻辑结束 ---
            elif model_type in ["SDNet", "SDNet70"]:
                output = model(images, model_dim)
                loss = F.mse_loss(output, labels, reduction='sum') / (model_dim[0] * model_dim[1] * train_batch_size)
            else:
                print(
                    'The "model_type" parameter selected in the train_for_one_stage(...)'
                    ' is the undefined network model keyword! Please check!')
                exit(0)

            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')

            # 损失反向传播
            loss.backward()

            # 参数更新
            optimizer.step()

            loss_of_epoch += loss.item()

            if iteration % display_step == 0:
                print('[{}] Epochs: {}/{}, Iteration: {}/{} --- Training Loss:{:.6f}'
                      .format(key_word, epoch + 1, cur_epochs, iteration, step * cur_epochs, loss.item()))

        ################################
        # 当前 epoch 结束
        ################################
        if (epoch + 1) % 1 == 0:
            # 计算当前 epoch 平均损失
            print('[{}] Epochs: {:d} finished ! Training loss: {:.5f}'
                  .format(key_word, epoch + 1, loss_of_epoch / max(1, batch_count)))

            # 将平均损失加入当前阶段数组
            loss_of_stage.append(loss_of_epoch / max(1, batch_count))

            # 统计当前 epoch 耗时
            time_elapsed = time.time() - cur_node_time
            print('[{}] Epochs consuming time: {:.0f}m {:.0f}s'
                  .format(key_word, time_elapsed // 60, time_elapsed % 60))
            training_time += time_elapsed
        #########################################################################
        # 达到中间结果保存点时... #
        #########################################################################
            if (epoch + 1) % save_epoch == 0:
                last_model_save_path = os.path.join(per_run_root, model_save_name + '_CurEpo' + str(epoch + 1) + '.pkl')
                torch.save(model.state_dict(), last_model_save_path)
                print('[' + key_word + '] Trained model saved: %d percent completed' % int((epoch + 1) * 100 / cur_epochs))

        np.save(os.path.join(per_run_root, "[Loss]" + model_save_name + ".npy"), np.array(loss_of_stage))

    return last_model_save_path, training_time


def curriculum_learning_training(model_type, init_model_src="", finetune_lr_scale=1.0):
    '''
    课程学习训练

    :param model_type:              主模型类型，根据不同论文实现进行区分。
                                    可用模型关键字为
                                    [DDNet70 | DDNet | InversionNet | FCNVMB| SDNet70 | SDNet]
    '''
    # Initialize training metadata (implicit mode based on load_pretrained)
    mode_dir = _training_mode_dir()
    print("[Train] mode_dir={}".format(mode_dir))

    # Write a simple run metadata file for traceability (implicit mode)
    run_base_root = os.path.join(results_dir, mode_dir, model_type, dataset_name)
    os.makedirs(run_base_root, exist_ok=True)
    run_meta_path = os.path.join(run_base_root, "run_meta.json")
    run_meta = {
        "mode_dir": mode_dir,
        "model_type": model_type,
        "load_pretrained": TRAIN_MANUAL_CONFIG.get("load_pretrained", ""),
        "finetune_lr_scale": finetune_lr_scale,
        "learning_rate": learning_rate,
        "train_size": train_size,
        "train_batch_size": train_batch_size,
        "firststage_epochs": firststage_epochs,
        "secondstage_epochs": secondstage_epochs,
        "thirdstage_epochs": thirdstage_epochs,
        "dataset": dataset_name,
        "inchannels": inchannels,
        "classes": classes,
    }
    try:
        with open(run_meta_path, 'w') as f:
            json.dump(run_meta, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    all_training_time = 0

    stage1_net_src = ""
    stage2_net_src = ""
    current_model_src = init_model_src
    effective_lr = learning_rate * finetune_lr_scale
    print("[Train] finetune_lr_scale={} effective_lr={}".format(finetune_lr_scale, effective_lr))
    if init_model_src:
        print("[Train] load pretrained checkpoint: {}".format(init_model_src))

    ###########
    # 阶段 1
    ###########
    if firststage_epochs != 0:
        print("read path: {}".format(current_model_src))
        net_model, device, optimizer = determine_network(
            external_model_src=current_model_src,
            model_type=model_type,
            lr=effective_lr,
        )
        training_loader, seismic_gathers, velocity_models = load_dataset(stage=1)
        stage1_net_src, training_time = train_for_one_stage(firststage_epochs, net_model, training_loader,
                                                            optimizer, key_word="CLStage1", model_type=model_type,
                                                            save_times=2)
        current_model_src = stage1_net_src
        all_training_time += training_time
        del training_loader
        del seismic_gathers
        del velocity_models
        del net_model
        del optimizer
        del device
        gc.collect()

    ###########
    # 阶段 2
    ###########
    if secondstage_epochs != 0:
        print("read path: {}".format(current_model_src))
        net_model, device, optimizer = determine_network(
            external_model_src=current_model_src,
            model_type=model_type,
            lr=effective_lr,
        )
        training_loader, seismic_gathers, velocity_models = load_dataset(stage=2)
        stage2_net_src, training_time = train_for_one_stage(secondstage_epochs, net_model, training_loader,
                                                            optimizer, key_word="CLStage2", model_type=model_type,
                                                            save_times=2)
        current_model_src = stage2_net_src
        all_training_time += training_time
        del training_loader
        del seismic_gathers
        del velocity_models
        del net_model
        del optimizer
        del device
        gc.collect()

    ###########
    # 阶段 3
    ###########
    if thirdstage_epochs != 0:
        print("read path: {}".format(current_model_src))
        net_model, device, optimizer = determine_network(
            external_model_src=current_model_src,
            model_type=model_type,
            lr=effective_lr,
        )
        training_loader, seismic_gathers, velocity_models = load_dataset(stage=3)
        stage3_net_src, training_time = train_for_one_stage(thirdstage_epochs, net_model, training_loader,
                                                            optimizer, key_word="CLStage3", model_type=model_type,
                                                            save_times=2)
        current_model_src = stage3_net_src
        all_training_time += training_time
        del training_loader
        del seismic_gathers
        del velocity_models
        del net_model
        del optimizer
        del device
        gc.collect()

    print("training runtime: {}s".format(all_training_time))


# 训练手动参数（不通过命令行传入）
TRAIN_MANUAL_CONFIG = {
    "model_type": "DDNet70",
    "load_pretrained": r"models_pretrain\CurveFaultAModel\DDNet70_SrcMix2_TgtCurveFaultA_PreEpo100_20260402_174016.pkl",
    "finetune_lr_scale": 0.1,
}

if __name__ == "__main__":
    if TRAIN_MANUAL_CONFIG["finetune_lr_scale"] <= 0:
        raise ValueError("--finetune-lr-scale must be > 0")

    curriculum_learning_training(
        model_type=TRAIN_MANUAL_CONFIG["model_type"],
        init_model_src=TRAIN_MANUAL_CONFIG["load_pretrained"],
        finetune_lr_scale=TRAIN_MANUAL_CONFIG["finetune_lr_scale"],
    )
    # main_train里的训练参数在param_config里更改
