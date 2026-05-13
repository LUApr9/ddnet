"""源域监督迁移预训练入口（M2）。

在不修改现有训练/测试脚本的前提下，提供独立预训练流程：

1) 解析源域/目标域预训练参数
2) 校验源域数据文件结构
3) 按源域顺序训练（同一模型权重连续继承）
4) 将最终权重保存到 models_pretrain/<target>Model/

说明：
- 复用现有网络构建与损失定义。
- 当前按 OpenFWI 风格读取 npy，默认要求 train_data/{seismic,vmodel}
  下存在 seismic*.npy 与 vmodel*.npy。
"""

import copy
import gc
import json
import os
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils

from func.datasets_reader import batch_read_npyfile
from func.device_selector import get_runtime_device
from func.utils import model_reader
from net.DDNet import DDNetModel, SDNetModel
from net.DDNet70 import DDNet70Model, SDNet70Model, LossDDNet
from net.FCNVMB import FCNVMB
from net.InversionNet import InversionNet
from param_config import (
    classes,
    data_dim,
    dataset_name,
    device_mode,
    display_step,
    inchannels,
    learning_rate,
    loss_weight,
    model_dim,
    model_type,
)


# 数据集名 -> 训练集分块数（每个分块通常约 500 条样本）
# 该映射同时用于文件完整性校验与分块读取范围。
SUPPORTED_OPENFWI_DATASETS = {
    "FlatVelA": 1,
    "CurveVelA": 1,
    "FlatFaultA": 1,
    "CurveFaultA": 1,
}


# 预训练手动参数
PRETRAIN_MANUAL_CONFIG = {
    "model_type": model_type,
    "source_datasets": "FlatVelA,CurveVelA",
    "target_dataset": "CurveFaultA",
    "pretrain_epochs": 100,
    "batch_size": 16,
    "lr": learning_rate,
    "save_name": "",
    "pretrained_path": "",
    "fallback_init_scratch": False,
    "strict_source_check": False,
    "nan_retry_scale": 0.1,
    "dry_run": False,
}


def parse_csv_list(value):
    """将逗号分隔字符串解析为列表，并校验非空。"""
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("source dataset list is empty")
    return items


def parse_args():
    """读取手动配置并做基础参数合法性校验。"""
    # 所有参数均从 PRETRAIN_MANUAL_CONFIG 读取。
    args = SimpleNamespace(**PRETRAIN_MANUAL_CONFIG)
    if args.pretrain_epochs < 0:
        raise ValueError("pretrain-epochs must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if args.lr <= 0:
        raise ValueError("lr must be > 0")
    if args.nan_retry_scale <= 0:
        raise ValueError("nan-retry-scale must be > 0")
    args.source_datasets = parse_csv_list(args.source_datasets)
    return args


def ensure_pretrain_dir(target_dataset):
    """创建并返回预训练输出目录。"""
    pretrain_dir = os.path.join("models_pretrain", "{}Model".format(target_dataset))
    os.makedirs(pretrain_dir, exist_ok=True)
    return pretrain_dir


def short_dataset_tag(source_datasets):
    """为源域列表生成简短标签（单源域名或 MixN）。"""
    if len(source_datasets) == 1:
        return source_datasets[0]
    return "Mix{}".format(len(source_datasets))


def build_output_basename(args):
    """构建输出文件基名；若未指定则自动带时间戳。"""
    if args.save_name:
        return args.save_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "{}_Src{}_Tgt{}_PreEpo{}_{}".format(
        args.model_type,
        short_dataset_tag(args.source_datasets),
        args.target_dataset,
        args.pretrain_epochs,
        timestamp,
    )


def resolve_dataset_dir(dataset):
    """根据数据集名拼接数据目录路径。"""
    return os.path.join("data", dataset) + os.sep


def validate_dataset_files(dataset):
    """检查源域数据是否完整 返回 (ok, reason) 。"""
    # 让主流程可按 strict 模式“立即失败”或按默认模式“跳过继续”。
    if dataset not in SUPPORTED_OPENFWI_DATASETS:
        return False, "unsupported_dataset"

    root = resolve_dataset_dir(dataset)
    seismic_dir = os.path.join(root, "train_data", "seismic")
    vmodel_dir = os.path.join(root, "train_data", "vmodel")

    if not os.path.isdir(seismic_dir) or not os.path.isdir(vmodel_dir):
        return False, "train_data_folders_not_found"

    expected_chunks = SUPPORTED_OPENFWI_DATASETS[dataset]
    missing = []
    for idx in range(1, expected_chunks + 1):
        se = os.path.join(seismic_dir, "seismic{}.npy".format(idx))
        vm = os.path.join(vmodel_dir, "vmodel{}.npy".format(idx))
        if not os.path.exists(se):
            missing.append(se)
        if not os.path.exists(vm):
            missing.append(vm)

    if missing:
        preview = "; ".join(missing[:3])
        return False, "missing_files:{} examples:{}".format(len(missing), preview)

    return True, "ok"


def determine_network(current_model_type, external_model_src="", lr=learning_rate):
    """创建网络、可选加载权重，并返回模型/设备/优化器。"""
    device, cuda_available, resolved_mode = get_runtime_device(device_mode)
    print("[Device] mode={} resolved={} cuda_available={}".format(resolved_mode, device.type, torch.cuda.is_available()))

    if current_model_type == "DDNet":
        net_model = DDNetModel(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    elif current_model_type == "DDNet70":
        net_model = DDNet70Model(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    elif current_model_type == "SDNet":
        net_model = SDNetModel(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    elif current_model_type == "SDNet70":
        net_model = SDNet70Model(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    elif current_model_type == "InversionNet":
        net_model = InversionNet()
    elif current_model_type == "FCNVMB":
        net_model = FCNVMB(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    else:
        raise ValueError("Undefined model_type: {}".format(current_model_type))

    if external_model_src:
        net_model = model_reader(net=net_model, device=device, save_src=external_model_src)

    if cuda_available:
        net_model = torch.nn.DataParallel(net_model.cuda(), device_ids=[0])
    else:
        net_model = net_model.to(device)

    optimizer = torch.optim.Adam(net_model.parameters(), lr=lr)
    return net_model, device, optimizer


def load_source_dataset(dataset, batch_size):
    """读取单个源域数据并封装为 DataLoader。"""
    dataset_dir = resolve_dataset_dir(dataset)
    file_chunks = SUPPORTED_OPENFWI_DATASETS[dataset]
    seismic, labels = batch_read_npyfile(dataset_dir, 1, file_chunks, "train")

    # Keep behavior aligned with model_train.py for OpenFWI labels normalization.
    for i in range(seismic.shape[0]):
        vm = labels[0][i][0]
        vm_min = np.min(vm)
        vm_max = np.max(vm)
        denom = vm_max - vm_min
        if denom < 1e-12:
            labels[0][i][0] = 0.0
        else:
            labels[0][i][0] = (vm - vm_min) / denom

    train_set = data_utils.TensorDataset(
        torch.from_numpy(seismic).float(),
        torch.from_numpy(labels[0]).float(),
        torch.from_numpy(labels[1]).long(),
    )
    loader = data_utils.DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True)
    return loader, seismic.shape[0]


def run_one_domain_training(
    model,
    optimizer,
    loader,
    total_samples,
    epochs,
    model_type_for_loss,
    log_prefix,
):
    """在一个源域上执行指定轮数训练。"""
    # 在当前模型/优化器状态下，对单个源域训练指定轮数。
    if epochs == 0:
        print("[{}] epochs=0, skip training".format(log_prefix))
        return

    model_device = next(model.parameters()).device
    steps_per_epoch = max(1, int(np.ceil(total_samples / float(loader.batch_size))))
    criterion = LossDDNet(weights=loss_weight)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (images, labels, contour_labels) in enumerate(loader):
            images = images.to(model_device, non_blocking=model_device.type == "cuda")
            labels = labels.to(model_device, non_blocking=model_device.type == "cuda")
            contour_labels = contour_labels.to(model_device, non_blocking=model_device.type == "cuda")

            optimizer.zero_grad()

            if model_type_for_loss in ["DDNet", "DDNet70"]:
                outputs = model(images, model_dim)
                loss = criterion(outputs[0], outputs[1], labels, contour_labels)
            elif model_type_for_loss in ["SDNet", "SDNet70"]:
                output = model(images, model_dim)
                loss = F.mse_loss(output, labels, reduction="sum") / (
                    model_dim[0] * model_dim[1] * images.shape[0]
                )
            else:
                raise ValueError("M2 training currently supports DDNet/SDNet family only")

            if np.isnan(float(loss.item())):
                raise ValueError("loss is nan while pretraining")

            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

            global_step = epoch * steps_per_epoch + step + 1
            if global_step % display_step == 0:
                print(
                    "[{}] epoch {}/{} step {}/{} loss={:.6f}".format(
                        log_prefix,
                        epoch + 1,
                        epochs,
                        global_step,
                        steps_per_epoch * epochs,
                        float(loss.item()),
                    )
                )

        elapsed = time.time() - t0
        mean_loss = epoch_loss / max(1, len(loader))
        print("[{}] epoch {} done mean_loss={:.6f} time={:.1f}s".format(log_prefix, epoch + 1, mean_loss, elapsed))


def train_one_domain_with_nan_retry(
    model,
    optimizer,
    loader,
    total_samples,
    epochs,
    model_type_for_loss,
    log_prefix,
    retry_lr_scale,
):
    """单源域训练包装器：遇到 NaN 时降学习率重试一次。"""
    # 单源域训练包装器：
    # 若出现 NaN 损失，则降低学习率并在同一源域重试一次。
    try:
        run_one_domain_training(
            model=model,
            optimizer=optimizer,
            loader=loader,
            total_samples=total_samples,
            epochs=epochs,
            model_type_for_loss=model_type_for_loss,
            log_prefix=log_prefix,
        )
        return optimizer, {"status": "ok", "error": ""}
    except Exception as err:
        err_msg = str(err)
        if "loss is nan" not in err_msg.lower():
            raise

        print("[{}] NaN detected, retry once with lr scale {}".format(log_prefix, retry_lr_scale))

        if hasattr(model, "module"):
            backup_state = copy.deepcopy(model.module.state_dict())
            model.module.load_state_dict(backup_state)
        else:
            backup_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(backup_state)

        current_lr = optimizer.param_groups[0]["lr"]
        retry_lr = current_lr * retry_lr_scale
        optimizer = torch.optim.Adam(model.parameters(), lr=retry_lr)

        run_one_domain_training(
            model=model,
            optimizer=optimizer,
            loader=loader,
            total_samples=total_samples,
            epochs=epochs,
            model_type_for_loss=model_type_for_loss,
            log_prefix=log_prefix + "-retry",
        )
        return optimizer, {"status": "ok_after_retry", "error": "nan_retry"}


def save_final_checkpoint(model, output_path):
    """保存最终模型权重（兼容 DataParallel）。"""
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, output_path)
    print("[Pretrain] saved checkpoint: {}".format(output_path))


def print_plan(args, output_path):
    """打印本次预训练计划与关键参数。"""
    print("=" * 72)
    print("[Pretrain] Source-domain supervised transfer plan")
    print("[Pretrain] model_type      : {}".format(args.model_type))
    print("[Pretrain] source_datasets : {}".format(", ".join(args.source_datasets)))
    print("[Pretrain] target_dataset  : {}".format(args.target_dataset))
    print("[Pretrain] pretrain_epochs : {} (per source domain)".format(args.pretrain_epochs))
    print("[Pretrain] batch_size      : {}".format(args.batch_size))
    print("[Pretrain] lr              : {}".format(args.lr))
    print("[Pretrain] init_ckpt       : {}".format(args.pretrained_path or "<none>"))
    print("[Pretrain] fallback_scratch: {}".format(args.fallback_init_scratch))
    print("[Pretrain] strict_check    : {}".format(args.strict_source_check))
    print("[Pretrain] nan_retry_scale : {}".format(args.nan_retry_scale))
    print("[Pretrain] output_ckpt     : {}".format(output_path))
    print("=" * 72)


def save_pretrain_summary(pretrain_dir, output_basename, summary):
    """将本次预训练过程摘要写入 JSON 文件。"""
    summary_path = os.path.join(pretrain_dir, output_basename + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[Pretrain] summary saved: {}".format(summary_path))
    return summary_path


def main():
    """预训练主流程调度入口。"""
    # 主流程：
    # 1) 解析参数并确定输出路径
    # 2) 校验并筛选源域（strict 模式失败，否则跳过）
    # 3) dry-run 仅输出 summary 后退出
    # 4) 初始化模型（支持加载预训练与失败回退）
    # 5) 逐个有效源域顺序训练
    # 6) 保存最终 checkpoint 与 summary 文件
    args = parse_args()
    pretrain_dir = ensure_pretrain_dir(args.target_dataset)
    output_basename = build_output_basename(args)
    output_ckpt = os.path.join(pretrain_dir, output_basename + ".pkl")

    print_plan(args, output_ckpt)

    summary = {
        "model_type": args.model_type,
        "target_dataset": args.target_dataset,
        "output_ckpt": output_ckpt,
        "source_requested": args.source_datasets,
        "source_skipped": [],
        "source_trained": [],
        "init_checkpoint": args.pretrained_path,
        "init_mode": "pretrained" if args.pretrained_path else "scratch",
    }

    valid_sources = []
    for source in args.source_datasets:
        ok, reason = validate_dataset_files(source)
        if ok:
            valid_sources.append(source)
            continue
        summary["source_skipped"].append({"dataset": source, "reason": reason})
        print("[Pretrain] skip source {}: {}".format(source, reason))
        if args.strict_source_check:
            raise RuntimeError("strict source check failed for {}: {}".format(source, reason))

    if not valid_sources:
        raise RuntimeError("no valid source dataset available for pretraining")

    if args.dry_run:
        print("[Pretrain] dry-run done. No training executed.")
        save_pretrain_summary(pretrain_dir, output_basename, summary)
        return

    init_ckpt = args.pretrained_path
    try:
        net_model, device, optimizer = determine_network(
            current_model_type=args.model_type,
            external_model_src=init_ckpt,
            lr=args.lr,
        )
    except Exception as err:
        if not init_ckpt or not args.fallback_init_scratch:
            raise
        print("[Pretrain] pretrained load failed, fallback to scratch: {}".format(err))
        summary["init_mode"] = "scratch_fallback"
        summary["init_error"] = str(err)
        net_model, device, optimizer = determine_network(
            current_model_type=args.model_type,
            external_model_src="",
            lr=args.lr,
        )

    for idx, source in enumerate(valid_sources, start=1):
        print("[Pretrain] ({}/{}) source domain: {}".format(idx, len(valid_sources), source))
        loader, sample_count = load_source_dataset(source, args.batch_size)
        optimizer, train_status = train_one_domain_with_nan_retry(
            model=net_model,
            optimizer=optimizer,
            loader=loader,
            total_samples=sample_count,
            epochs=args.pretrain_epochs,
            model_type_for_loss=args.model_type,
            log_prefix="{}:{}".format(source, args.model_type),
            retry_lr_scale=args.nan_retry_scale,
        )
        summary["source_trained"].append(
            {
                "dataset": source,
                "sample_count": int(sample_count),
                "status": train_status.get("status", "ok"),
                "error": train_status.get("error", ""),
            }
        )
        del loader
        gc.collect()

    save_final_checkpoint(net_model, output_ckpt)
    save_pretrain_summary(pretrain_dir, output_basename, summary)

    del net_model
    del optimizer
    del device
    gc.collect()


if __name__ == "__main__":
    main()
