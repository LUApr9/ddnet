# 地震全波形反演预测系统 (DD-Net70) - 实验操作指南

本项目实现了基于 **DD-Net70** 架构的地震全波形反演系统，集成了**课程学习 (CL)**、**监督迁移预训练 (M2)** 与 **自监督掩码自动编码器预训练 (MSAE)** 策略。

---

## 1. 环境准备

1. **进入项目目录**:
   ```bash
   cd D:\bishe\ddnet-master
   ```

2. **确认依赖环境**:
   ```bash
   python -c "import torch, numpy, scipy, cv2, matplotlib"
   ```

3. **路径初始化**:
   打开 `path_config.py`，确保 `main_dir = 'D:/bishe/ddnet-master/'` 指向项目实际根目录。
   
---

## 2. 数据准备

请将数据集（以 `CurveFaultA` 为例）按以下结构放置：

* **训练集**:
* `data/CurveFaultA/train_data/seismic/` (放置 `seismic*.npy`)
* `data/CurveFaultA/train_data/vmodel/` (放置 `vmodel*.npy`)


* **测试集**: 放入 `data/CurveFaultA/test_data/` 对应子目录下。

---

## 3. 实验路径：监督训练 (主训练)

### **A. 从头训练 (From-Scratch)**

1. 打开 `model_train.py`，在底部的 `TRAIN_MANUAL_CONFIG` 中设置：
* `"load_pretrained": ""` (**必须留空**)

2. 运行训练：
```bash
python model_train.py

```


3. **输出**: 模型与实验日志将自动保存在 `results/unused_pretrain/`。

### **B. 监督迁移学习微调 (M2)**

1. **源域预训练**: 运行 `python pretrain_entry.py` 生成跨域迁移权重。
2. **目标域微调**: 打开 `model_train.py`，修改 `TRAIN_MANUAL_CONFIG`：
* `"load_pretrained"`: 设置为 `models_pretrain/CurveFaultAModel/xxx.pkl`
* `"finetune_lr_scale"`: 建议设为 `0.1` (以较小学习率微调)


3. 运行微调：
```bash
python model_train.py

```

4. **输出**: 结果将自动分流保存至 `results/used_pretrain/`。

---

## 4. 实验路径：自监督预训练 (MSAE)

### **第一步：特征提取预训练**

1. 打开 `selfsup_msae_pretrain.py`，在 `MANUAL_CONFIG` 中确认 `source_datasets` 列表。
2. 运行：
```bash
python selfsup_msae_pretrain.py
```

3. **产出**: 骨干网权重保存在 `models_pretrain/CurveFaultAModel/xxx_backbone.pth`。

### **第二步：下游任务微调**

1. 打开 `selfsup_to_ddnet70_finetune.py`，设置：
* `"pretrained_path"`: 指向上一步生成的 `.pth` 权重文件。

2. 运行：
```bash
python selfsup_to_ddnet70_finetune.py
```

3. **输出**: 微调后的模型保存在 `models_selfsup/CurveFaultAModel/`。

---

## 5. 测试与多模型评估

1. **配置对比项**: 编辑 `compare_models.pretrain.json`，在 `models` 列表中填入各实验生成的权重路径。
2. **启动测试**: 修改 `model_test.py` 底部配置：
```python
TEST_MANUAL_CONFIG = {
    "compare_config": "compare_models.pretrain.json",
    "enable_lpips": False
}

```

3. **运行测试**:
```bash
python model_test.py

```

4. **查看结果**:
* **定量指标**: `results/CurveFaultAResults/[CompareBatch]...csv`
* **定性预测图**: `results/CurveFaultAResults/previews/`

---

## 6. 可视化演示系统

启动基于 **PyQt5** 的图形化界面，支持手动加载模型进行单样本反演实时展示：

```bash
python -m demo.qt_host_demo
```
