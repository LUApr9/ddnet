"""地震自监督预训练（MSAE）独立脚本。

说明：
1) 不依赖现有训练入口（model_train.py / pretrain_entry.py）。
2) 仅使用 seismic 数据做掩码重建，不使用 vmodel 标签。
3) 产出 checkpoint 与 summary，便于后续与监督预训练方案对比。
"""

import gc
import json
import os
import random
import time
from bisect import bisect_right
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from net.DDNet70 import SeismicRecordDownSampling, unetDown
from param_config import dataset_name, learning_rate, device_mode

# 手动配置区（不使用命令行参数）
MANUAL_CONFIG = {
    # 数据配置
    "source_datasets": ["FlatVelA", "CurveVelA", "FlatFaultA"],
    "data_root": "data",
    "chunk_limit_per_domain": 0,  # 0 表示使用该域全部 seismic*.npy
    "batch_size": 16,
    "num_workers": 0,

    # 训练配置
    "epochs": 100,
    "steps_per_epoch": 100,
    "learning_rate": learning_rate,
    "weight_decay": 1e-6,
    "seed": 42,

    # 掩码配置（块掩码）
    "mask_ratio": 0.6,
    "mask_block_h": 16,
    "mask_block_w": 8,

    # 输出配置
    "target_tag": dataset_name,
    "save_name": "",  # 留空则自动命名
    "save_dir": "models_pretrain",
    "save_every_epoch": 0,  # 0 表示不保存中间权重

    # 运行控制
    "device_mode": device_mode,
    "dry_run": False,
}


# 函数作用：固定随机种子，尽量提升复现性。
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 函数作用：根据配置解析运行设备。
def resolve_device(device_mode):
    mode = str(device_mode).lower().strip()
    if mode == "cpu":
        return torch.device("cpu"), "cpu"
    if mode == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("device_mode=gpu 但当前不可用 CUDA")
        return torch.device("cuda"), "gpu"

    if torch.cuda.is_available():
        return torch.device("cuda"), "auto->gpu"
    return torch.device("cpu"), "auto->cpu"


# 函数作用：按数字后缀排序 seismic*.npy 文件路径。
def sorted_seismic_files(domain_dir):
    seismic_dir = os.path.join(domain_dir, "train_data", "seismic")
    if not os.path.isdir(seismic_dir):
        return []

    files = []
    for name in os.listdir(seismic_dir):
        if name.startswith("seismic") and name.endswith(".npy"):
            files.append(os.path.join(seismic_dir, name))

    # 函数作用：提取 seismic 文件名中的数字后缀用于排序。
    def extract_idx(path):
        base = os.path.splitext(os.path.basename(path))[0]
        num = base.replace("seismic", "")
        return int(num) if num.isdigit() else 10**9

    files.sort(key=extract_idx)
    return files


# 函数作用：惰性读取多块 seismic 文件，按样本索引返回单条 seismic。
class SeismicChunkDataset(Dataset):
    # 函数作用：初始化分块数据集，记录每个分块样本数与累计索引。
    def __init__(self, chunk_paths):
        if not chunk_paths:
            raise ValueError("chunk_paths 为空，无法构建数据集")

        self.chunk_paths = list(chunk_paths)
        self._memmaps = {}
        self._sizes = []
        self._cum = []

        total = 0
        for p in self.chunk_paths:
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 4:
                raise ValueError("期望 seismic 形状为 [N,C,H,W]，实际 {} from {}".format(arr.shape, p))
            n = int(arr.shape[0])
            if n <= 0:
                raise ValueError("空分块文件: {}".format(p))
            self._sizes.append(n)
            total += n
            self._cum.append(total)

    # 函数作用：返回数据集总样本数。
    def __len__(self):
        return self._cum[-1]

    # 函数作用：按分块索引惰性读取 memmap，避免一次性占满内存。
    def _get_chunk(self, chunk_idx):
        path = self.chunk_paths[chunk_idx]
        mm = self._memmaps.get(path)
        if mm is None:
            mm = np.load(path, mmap_mode="r")
            self._memmaps[path] = mm
        return mm

    # 函数作用：按全局样本索引取出单条 seismic 并做标准化。
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")

        chunk_idx = bisect_right(self._cum, idx)
        prev_cum = 0 if chunk_idx == 0 else self._cum[chunk_idx - 1]
        local_idx = idx - prev_cum
        arr = self._get_chunk(chunk_idx)
        sample = np.array(arr[local_idx], dtype=np.float32, copy=True)

        # 每样本标准化，避免不同源域振幅尺度差异过大。
        mean = float(sample.mean())
        std = float(sample.std())
        if std < 1e-6:
            sample = sample - mean
        else:
            sample = (sample - mean) / std

        return torch.from_numpy(sample)


# 函数作用：基于 DDNet70 编码主干的掩码重建模型。
class DDNet70BackboneMSAE(nn.Module):
    # 函数作用：构建 DDNet70 同构编码器 + 轻量重建头。
    def __init__(self, in_channels):
        super().__init__()
        # DDNet70 编码主干（用于后续微调初始化迁移）。
        self.pre_seis_conv = SeismicRecordDownSampling(in_channels)
        self.down3 = unetDown(32, 64, True)
        self.down4 = unetDown(64, 128, True)
        self.down5 = unetDown(128, 256, True)
        self.center = unetDown(256, 512, True)

        # 自监督重建头（不参与 DDNet70 微调加载）。
        self.recon_up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.recon_up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.recon_up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.recon_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
        )

    # 函数作用：前向传播，输出重建后的 seismic。
    def forward(self, x):
        feat = self.pre_seis_conv(x)
        feat = self.down3(feat)
        feat = self.down4(feat)
        feat = self.down5(feat)
        feat = self.center(feat)

        recon = self.recon_up3(feat)
        recon = self.recon_up2(recon)
        recon = self.recon_up1(recon)
        out = self.recon_head(recon)

        # 强制重建输出与输入空间尺寸一致（避免 72 vs 70）
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out


# 函数作用：导出可被 DDNet70 微调脚本加载的编码主干参数。
def export_ddnet70_backbone_state(msae_state):
    keep_prefixes = ("pre_seis_conv.", "down3.", "down4.", "down5.", "center.")
    out = {}
    for k, v in msae_state.items():
        if k.startswith(keep_prefixes):
            out[k] = v
    return out


# 函数作用：生成块掩码并返回掩码输入与 mask（二值，1 代表被遮挡区域）。
def apply_block_mask(x, mask_ratio, block_h, block_w):
    b, c, h, w = x.shape
    mask = torch.zeros((b, 1, h, w), dtype=x.dtype, device=x.device)

    block_area = max(1, block_h * block_w)
    target_area = int(mask_ratio * h * w)
    num_blocks = max(1, target_area // block_area)

    max_top = max(1, h - block_h + 1)
    max_left = max(1, w - block_w + 1)

    for bi in range(b):
        for _ in range(num_blocks):
            top = int(torch.randint(0, max_top, (1,), device=x.device).item())
            left = int(torch.randint(0, max_left, (1,), device=x.device).item())
            bottom = min(h, top + block_h)
            right = min(w, left + block_w)
            mask[bi, 0, top:bottom, left:right] = 1.0

    x_masked = x * (1.0 - mask)
    return x_masked, mask


# 函数作用：仅在掩码区域计算重建损失。
def masked_l1_loss(recon, target, mask):
    valid = mask.expand_as(target)
    denom = valid.sum().clamp_min(1.0)
    return (torch.abs(recon - target) * valid).sum() / denom


# 函数作用：构建每个源域的数据加载器。
def build_domain_loaders(cfg):
    root = cfg["data_root"]
    limit = int(cfg["chunk_limit_per_domain"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])

    domain_loaders = {}
    sample_shape = None

    for domain in cfg["source_datasets"]:
        domain_dir = os.path.join(root, domain)
        files = sorted_seismic_files(domain_dir)
        if limit > 0:
            files = files[:limit]
        if not files:
            print("[MSAE] 跳过域 {}: 未发现 seismic*.npy".format(domain))
            continue

        ds = SeismicChunkDataset(files)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        domain_loaders[domain] = loader

        if sample_shape is None:
            sample_shape = tuple(ds[0].shape)

        print("[MSAE] 域 {}: chunks={} samples={}".format(domain, len(files), len(ds)))

    if not domain_loaders:
        raise RuntimeError("没有可用源域数据，请检查 data_root 和 source_datasets")

    return domain_loaders, sample_shape


# 函数作用：生成保存文件基名。
def build_save_basename(cfg):
    if cfg["save_name"]:
        return cfg["save_name"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "MSAE_Src{}_Tgt{}_Epo{}_{}".format(
        len(cfg["source_datasets"]),
        cfg["target_tag"],
        cfg["epochs"],
        stamp,
    )


# 函数作用：执行自监督预训练主循环。
def train_msae(cfg):
    set_seed(int(cfg["seed"]))
    device, resolved = resolve_device(cfg["device_mode"])
    print("[MSAE] device_mode={} resolved={}".format(cfg["device_mode"], resolved))

    domain_loaders, sample_shape = build_domain_loaders(cfg)
    in_channels = int(sample_shape[0])

    model = DDNet70BackboneMSAE(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    domains = list(domain_loaders.keys())
    iterators = {k: iter(v) for k, v in domain_loaders.items()}

    save_dir = os.path.join(cfg["save_dir"], "{}Model".format(cfg["target_tag"]))
    os.makedirs(save_dir, exist_ok=True)
    base = build_save_basename(cfg)
    ckpt_path = os.path.join(save_dir, base + ".pth")
    ckpt_full_path = os.path.join(save_dir, base + "_full_msae.pth")
    ckpt_backbone_path = os.path.join(save_dir, base + "_ddnet70_backbone.pth")
    summary_path = os.path.join(save_dir, base + "_summary.json")

    if cfg["dry_run"]:
        summary = {
            "mode": "dry_run",
            "domains": domains,
            "in_channels": in_channels,
            "sample_shape": sample_shape,
            "save_path": ckpt_path,
            "checkpoint_full_msae": ckpt_full_path,
            "checkpoint_ddnet70_backbone": ckpt_backbone_path,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[MSAE] dry-run 完成，summary: {}".format(summary_path))
        return

    history = []
    global_step = 0
    t_start = time.time()

    for epoch in range(int(cfg["epochs"])):
        model.train()
        epoch_loss = 0.0
        epoch_steps = int(cfg["steps_per_epoch"])
        epoch_t0 = time.time()

        for step in range(epoch_steps):
            domain = domains[step % len(domains)]
            it = iterators[domain]
            try:
                batch = next(it)
            except StopIteration:
                it = iter(domain_loaders[domain])
                iterators[domain] = it
                batch = next(it)

            x = batch.to(device, non_blocking=(device.type == "cuda"))
            x_masked, mask = apply_block_mask(
                x,
                mask_ratio=float(cfg["mask_ratio"]),
                block_h=int(cfg["mask_block_h"]),
                block_w=int(cfg["mask_block_w"]),
            )

            optimizer.zero_grad()
            recon = model(x_masked)
            loss = masked_l1_loss(recon, x, mask)

            if not torch.isfinite(loss):
                raise RuntimeError("出现非有限损失（NaN/Inf），请降低学习率或掩码比例")

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            global_step += 1

            if global_step % 20 == 0:
                print(
                    "[MSAE] epoch {}/{} step {}/{} domain={} loss={:.6f}".format(
                        epoch + 1,
                        cfg["epochs"],
                        step + 1,
                        epoch_steps,
                        domain,
                        float(loss.item()),
                    )
                )

        mean_loss = epoch_loss / max(1, epoch_steps)
        epoch_time = time.time() - epoch_t0
        history.append({"epoch": epoch + 1, "mean_loss": mean_loss, "seconds": epoch_time})
        print("[MSAE] epoch {} done mean_loss={:.6f} time={:.1f}s".format(epoch + 1, mean_loss, epoch_time))

        save_every = int(cfg["save_every_epoch"])
        if save_every > 0 and (epoch + 1) % save_every == 0:
            mid_path = os.path.join(save_dir, "{}_E{}.pth".format(base, epoch + 1))
            torch.save(model.state_dict(), mid_path)
            print("[MSAE] saved mid checkpoint: {}".format(mid_path))

    elapsed = time.time() - t_start
    final_state = model.state_dict()
    backbone_state = export_ddnet70_backbone_state(final_state)

    # 兼容历史流程：保留 base.pth 路径并保存全量 MSAE 权重。
    torch.save(final_state, ckpt_path)
    # print("[MSAE][MSAE_FULL] saved (legacy): {}".format(ckpt_path))

    torch.save(final_state, ckpt_full_path)
    # print("[MSAE][MSAE_FULL] saved: {}".format(ckpt_full_path))

    torch.save(backbone_state, ckpt_backbone_path)
    print("[MSAE][DDNET70_BACKBONE] saved: {} (keys={})".format(ckpt_backbone_path, len(backbone_state)))

    summary = {
        "strategy": "MSAE",
        "config": cfg,
        "device": str(device),
        "domains": domains,
        "in_channels": in_channels,
        "sample_shape": sample_shape,
        "checkpoint": ckpt_path,
        "checkpoint_full_msae": ckpt_full_path,
        "checkpoint_ddnet70_backbone": ckpt_backbone_path,
        "total_seconds": elapsed,
        "final_mean_loss": history[-1]["mean_loss"] if history else None,
        "history": history,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[MSAE] saved summary: {}".format(summary_path))

    del model
    del optimizer
    gc.collect()


# 函数作用：脚本入口。
def main():
    train_msae(MANUAL_CONFIG)


if __name__ == "__main__":
    main()
