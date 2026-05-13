"""将自监督预训练权重用于 DDNet 系列微调（独立脚本）。

用途：
1) 独立于既有入口，通过文件内参数控制微调。
2) 在启动前检查预训练权重与目标网络是否完全兼容。
3) 不兼容时可按配置回退为从头训练（避免流程中断）。

说明：
- 本脚本复用 model_train.py 的课程学习训练流程。
- 该脚本只负责“选择是否加载预训练权重 + 回退策略”。
"""

import os

import torch

import model_train as train_module
from model_train import curriculum_learning_training
from net.DDNet import DDNetModel, SDNetModel
from net.DDNet70 import DDNet70Model, SDNet70Model
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

# 手动配置区（不使用命令行参数）
MANUAL_CONFIG = {
    "model_type": model_type,
    "pretrained_path": "models_pretrain\CurveFaultAModel\MSAE_Src3_TgtCurveFaultA_Epo100_20260404_164310_ddnet70_backbone.pth",  # 例如：models_pretrain/CurveFaultAModel/xxx.pth 或 xxx.pkl
    "pretrained_load_mode": "partial_backbone",  # strict_full | partial_backbone
    "finetune_lr_scale": 0.1,
    "strict_pretrained_match": False,   # True: 不兼容直接报错
    "allow_scratch_fallback": False,     # True: 不兼容时从头训练
    "models_root": "models_selfsup",
}


# 函数作用：按 model_type 创建未加载权重的网络骨架。
def build_model_skeleton(model_type):
    if model_type == "DDNet":
        return DDNetModel(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    if model_type == "DDNet70":
        return DDNet70Model(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    if model_type == "SDNet":
        return SDNetModel(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    if model_type == "SDNet70":
        return SDNet70Model(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    if model_type == "InversionNet":
        return InversionNet()
    if model_type == "FCNVMB":
        return FCNVMB(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    raise ValueError("未支持的 model_type: {}".format(model_type))


# 函数作用：标准化 checkpoint 的键名（去掉 DataParallel 前缀 module.）。
def normalize_state_dict_keys(state):
    out = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


# 函数作用：将可能较长的键列表裁剪为可读字符串。
def format_key_preview(keys, max_items=8):
    keys = list(keys)
    if not keys:
        return "[]"
    shown = keys[:max_items]
    text = ", ".join(shown)
    if len(keys) > max_items:
        text += ", ..."
    return "[{}]".format(text)


# 函数作用：按配置过滤允许加载的参数前缀（用于主干部分加载）。
def get_allowed_prefixes(load_mode):
    if load_mode == "partial_backbone":
        return ("pre_seis_conv.", "down3.", "down4.", "down5.", "center.")
    return ()


# 函数作用：检查 checkpoint 在部分加载模式下的兼容性并统计加载信息。
def check_partial_compatibility(model_type, pretrained_path, load_mode):
    if not os.path.exists(pretrained_path):
        return False, "pretrained_file_not_found:{}".format(os.path.abspath(pretrained_path)), None

    try:
        raw_state = torch.load(pretrained_path, map_location="cpu")
    except Exception as e:
        return False, "checkpoint_load_error:{}:{}".format(type(e).__name__, e), None

    if not isinstance(raw_state, dict):
        return False, "checkpoint_not_state_dict:type={}".format(type(raw_state).__name__), None
    if not raw_state:
        return False, "checkpoint_empty_dict", None

    state = normalize_state_dict_keys(raw_state)
    model = build_model_skeleton(model_type)
    model_state = model.state_dict()
    allowed_prefixes = get_allowed_prefixes(load_mode)

    allowed_model_keys = []
    if allowed_prefixes:
        for k in model_state.keys():
            if k.startswith(allowed_prefixes):
                allowed_model_keys.append(k)
    else:
        allowed_model_keys = list(model_state.keys())

    loadable = {}
    missing = []
    shape_mismatch = []
    for k in allowed_model_keys:
        if k not in state:
            missing.append(k)
            continue
        if tuple(model_state[k].shape) != tuple(state[k].shape):
            shape_mismatch.append((k, tuple(state[k].shape), tuple(model_state[k].shape)))
            continue
        loadable[k] = state[k]

    unexpected = []
    allowed_key_set = set(allowed_model_keys)
    for k in state.keys():
        if k not in allowed_key_set:
            unexpected.append(k)

    stats = {
        "allowed_total": len(allowed_model_keys),
        "loadable_total": len(loadable),
        "missing_total": len(missing),
        "shape_mismatch_total": len(shape_mismatch),
        "unexpected_total": len(unexpected),
        "missing_preview": format_key_preview(missing),
        "unexpected_preview": format_key_preview(unexpected),
        "shape_mismatch_preview": shape_mismatch[:5],
        "loadable_state": loadable,
    }

    if len(loadable) == 0:
        return False, "no_loadable_keys_in_partial_mode", stats

    return True, "ok", stats


# 函数作用：检查 checkpoint 与目标模型是否“完全兼容可直接 load_state_dict”。
def check_full_compatibility(model_type, pretrained_path):
    if not os.path.exists(pretrained_path):
        return False, "pretrained_file_not_found:{}".format(os.path.abspath(pretrained_path))

    try:
        state = torch.load(pretrained_path, map_location="cpu")
    except Exception as e:
        return False, "checkpoint_load_error:{}:{}".format(type(e).__name__, e)

    if not isinstance(state, dict):
        return False, "checkpoint_not_state_dict:type={}".format(type(state).__name__)

    if not state:
        return False, "checkpoint_empty_dict"

    # 常见情况：checkpoint 包含 state_dict/model 等外层字段，提示排查方向。
    first_value = next(iter(state.values()))
    if isinstance(first_value, dict):
        top_keys = list(state.keys())
        return False, "checkpoint_nested_dict:top_keys={}".format(format_key_preview(top_keys))

    state = normalize_state_dict_keys(state)
    model = build_model_skeleton(model_type)
    model_state = model.state_dict()

    if len(state) != len(model_state):
        return False, "key_count_mismatch:ckpt={},model={}".format(len(state), len(model_state))

    for k, v in model_state.items():
        if k not in state:
            ckpt_preview = format_key_preview(state.keys())
            return False, "missing_key:{};ckpt_keys_preview={}".format(k, ckpt_preview)
        if tuple(v.shape) != tuple(state[k].shape):
            return False, "shape_mismatch:{};ckpt_shape={};model_shape={}".format(
                k, tuple(state[k].shape), tuple(v.shape)
            )

    return True, "ok"


# 函数作用：根据兼容性与策略决定最终使用的初始化权重路径。
def resolve_init_checkpoint(cfg):
    path = cfg["pretrained_path"].strip()
    load_mode = cfg.get("pretrained_load_mode", "strict_full").strip()
    if not path:
        print("[Finetune] 未指定 pretrained_path，将从头训练。")
        return ""

    if load_mode not in ["strict_full", "partial_backbone"]:
        raise ValueError("pretrained_load_mode 仅支持 strict_full | partial_backbone")

    print("[Finetune] 目标模型类型: {}".format(cfg["model_type"]))
    print("[Finetune] 加载模式: {}".format(load_mode))
    print("[Finetune] 预训练权重路径(原始): {}".format(path))
    print("[Finetune] 预训练权重路径(绝对): {}".format(os.path.abspath(path)))

    if load_mode == "partial_backbone":
        ok, reason, stats = check_partial_compatibility(cfg["model_type"], path, load_mode)
        if ok:
            print("[Finetune] 部分加载兼容，加载: {}".format(path))
            print(
                "[Finetune][LOAD_STATS] allowed={} loaded={} missing={} shape_mismatch={} unexpected={}".format(
                    stats["allowed_total"],
                    stats["loadable_total"],
                    stats["missing_total"],
                    stats["shape_mismatch_total"],
                    stats["unexpected_total"],
                )
            )
            if stats["missing_total"] > 0:
                print("[Finetune][LOAD_STATS] missing preview: {}".format(stats["missing_preview"]))
            if stats["shape_mismatch_total"] > 0:
                print("[Finetune][LOAD_STATS] shape mismatch preview: {}".format(stats["shape_mismatch_preview"]))
            if stats["unexpected_total"] > 0:
                print("[Finetune][LOAD_STATS] unexpected preview: {}".format(stats["unexpected_preview"]))

            # 为兼容现有训练入口（其内部使用 strict load_state_dict），
            # 将可加载主干参数合并到随机初始化模型并保存为临时完整权重。
            merged_model = build_model_skeleton(cfg["model_type"])
            merged_state = merged_model.state_dict()
            merged_state.update(stats["loadable_state"])
            merged_path = os.path.splitext(path)[0] + "_merged_for_finetune.pth"
            torch.save(merged_state, merged_path)
            print("[Finetune] 已生成兼容完整权重: {}".format(merged_path))
            return merged_path

        msg = "[Finetune] 预训练权重不兼容: {} ({})".format(path, reason)
        if stats is not None:
            msg += " | allowed={} loaded={} missing={} shape_mismatch={} unexpected={}".format(
                stats["allowed_total"],
                stats["loadable_total"],
                stats["missing_total"],
                stats["shape_mismatch_total"],
                stats["unexpected_total"],
            )
        if cfg["strict_pretrained_match"]:
            raise RuntimeError(msg)
        print(msg)
        if cfg["allow_scratch_fallback"]:
            print("[Finetune] 回退策略启用：改为从头训练。")
            return ""
        raise RuntimeError("预训练权重不兼容且未允许回退。")

    ok, reason = check_full_compatibility(cfg["model_type"], path)
    if ok:
        print("[Finetune] 预训练权重兼容，加载: {}".format(path))
        return path

    msg = "[Finetune] 预训练权重不兼容: {} ({})".format(path, reason)
    if cfg["strict_pretrained_match"]:
        raise RuntimeError(msg)

    print(msg)
    if cfg["allow_scratch_fallback"]:
        print("[Finetune] 回退策略启用：改为从头训练。")
        return ""

    raise RuntimeError("预训练权重不兼容且未允许回退。")


# 函数作用：脚本入口，执行微调训练。
def main():
    cfg = MANUAL_CONFIG
    if cfg["finetune_lr_scale"] <= 0:
        raise ValueError("finetune_lr_scale 必须 > 0")

    target_results_dir = os.path.join("result", "{}result".format(dataset_name), "SelfSup")
    os.makedirs(target_results_dir, exist_ok=True)
    train_module.results_dir = target_results_dir
    print("[Finetune] 输出结果目录: {}".format(target_results_dir))

    models_root = cfg.get("models_root", "models")
    target_models_dir = os.path.join(models_root, "{}Model".format(dataset_name))
    os.makedirs(target_models_dir, exist_ok=True)
    train_module.models_dir = target_models_dir
    print("[Finetune] 输出模型目录: {}".format(target_models_dir))

    init_ckpt = resolve_init_checkpoint(cfg)
    curriculum_learning_training(
        model_type=cfg["model_type"],
        init_model_src=init_ckpt,
        finetune_lr_scale=cfg["finetune_lr_scale"],
    )


if __name__ == "__main__":
    main()
