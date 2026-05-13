# DDNet 预训练策略设计（同域监督迁移，V1）

## 1. 背景与目标

当前仓库已经具备完整的训练与测试链路（`model_train.py`、`model_test.py`），并新增了预训练入口骨架（`pretrain_entry.py`）。本设计目标是在不破坏现有流程的前提下，落地一个可快速验证的预训练方案：

- 路线：同域监督迁移（source pretrain + target finetune）
- 目标：在 `CurveFaultA` 上相较无预训练基线，降低 `mse_mean`
- 约束：最小改动、快速验证优先、可回退

## 2. 现状分析（基于代码）

- 框架：PyTorch，模型选择集中在 `model_train.py:determine_network`
- 网络：`DDNet70` 支持双解码器联合损失（速度回归 + 轮廓分支）
- 数据：`func/datasets_reader.py` 已打通 `.mat/.npy` 读取，OpenFWI 数据格式可直接复用
- 训练：已有课程学习 stage1/2/3，可作为目标域微调机制
- 预训练：`pretrain_entry.py` 当前仅流程占位，不执行真实训练

> 备注：当前目录无 `.git`，无法记录“最近提交”信息。

## 3. 候选方案与取舍

### 方案 A（推荐）：同域监督迁移

- 做法：先在 `FlatVelA + CurveVelA` 预训练 `DDNet70`，再在 `CurveFaultA` 微调
- 优点：改动最小，工程风险低，符合快速验证
- 风险：源域与目标域差异过大时收益不稳定

### 方案 B：课程式多源预训练（easy -> hard）

- 做法：按难度顺序进行多源阶段预训练再迁移
- 优点：训练更稳，符合 curriculum 理念
- 风险：难度定义与阶段策略需要额外实验

### 方案 C：自监督预训练 + 监督微调

- 做法：先做 seismic encoder 自监督，再接反演头
- 优点：泛化潜力高
- 风险：实现复杂，不适合当前“快速验证”周期

结论：V1 选择方案 A。

## 4. 设计方案（V1）

### 4.1 架构与职责

- `pretrain_entry.py`：预训练调度器（源域列表、轮次、保存）
- `model_train.py`：目标域训练入口（可选加载预训练权重）
- `func/datasets_reader.py`：继续复用，不做核心改写
- 目录约束：
  - 预训练权重仅写入 `models_pretrain/...`
  - 正式实验权重继续写入 `models/...`

### 4.2 数据流

1. 读取源域列表（默认 `FlatVelA,CurveVelA`）
2. 顺序执行源域监督训练，持续更新同一模型权重
3. 产出预训练权重到 `models_pretrain/<tag>.pkl`
4. 在目标域 `CurveFaultA` 加载预训练权重并微调
5. 使用 `model_test.py --compare-config` 统一评估

### 4.3 训练与评估约束

- 固定模型：`DDNet70`
- 快速验证：小样本 + 少 epoch 先冒烟
- 主指标：`mse_mean`（唯一主排序）
- 对照组：
  - Baseline：无预训练直接训练
  - Proposed：预训练 + 微调

## 5. 错误处理与回退

- 数据缺失：单源域失败标记并跳过，不中断全流程
- 权重不兼容：回退从头训练并记录层级差异
- 训练 NaN：触发降学习率重试一次；仍失败则中止并保存诊断
- 设备不可用：`auto` 回退 CPU，并打印实际设备
- 回退路径：任意失败可退回原链路（`model_train.py` + `model_test.py`）

## 6. 验收标准

- 主标准：在相同测试集上，`Proposed` 的 `mse_mean` 低于 `Baseline`
- 稳定性建议：至少 3 个随机种子，>= 2 次优于基线
- 公平性：相同结构、相同测试脚本、相同数据切分与配置

## 7. V1 默认参数建议

- `model_type=DDNet70`
- `source_datasets=FlatVelA,CurveVelA`
- `target_dataset=CurveFaultA`
- `pretrain_epochs=5~10`（快速验证）
- `finetune_lr_scale=0.1`

## 8. 非目标（YAGNI）

- 不在 V1 引入自监督、蒸馏、复杂多任务重构
- 不改变现有 `model_test.py` 指标体系

## 9. 后续实施入口

本设计确认后，下一步应进入实现计划编写（writing-plans），将任务拆分为可执行步骤与验证清单。
