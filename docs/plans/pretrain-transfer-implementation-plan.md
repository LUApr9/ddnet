# DDNet 同域监督迁移预训练 V1 实施计划

## 1. 实施目标

在不破坏现有训练与测试流程的前提下，实现：

1. 源域监督预训练（`FlatVelA + CurveVelA`）
2. 目标域微调（`CurveFaultA`）
3. 基于 `mse_mean` 的统一对照评估

主约束：最小改动、快速验证优先、可回退。

## 2. 交付范围

### 2.1 In Scope

- 将 `pretrain_entry.py` 从骨架升级为可执行预训练调度
- 在正式训练入口提供“可选加载预训练权重”能力
- 预训练权重与实验权重目录隔离
- 增加最小运行文档与命令示例

### 2.2 Out of Scope

- 自监督预训练
- 蒸馏训练
- 大规模超参数搜索

## 3. 里程碑与任务拆分

### M1: 参数与路径打通

任务：

1. 在 `pretrain_entry.py` 新增参数：
   - `--source-datasets`（逗号分隔）
   - `--target-dataset`
   - `--pretrain-epochs`
   - `--finetune-epochs`（可选）
   - `--pretrained-path`（用于仅微调场景）
2. 增加 `models_pretrain/<tag>Model/` 目录生成逻辑
3. 统一预训练产物命名规范（包含模型、源域、epoch、时间戳）

完成标准：

- dry-run 可打印完整配置与产物路径

---

### M2: 源域预训练循环

任务：

1. 复用现有模型构建与训练函数（避免重写训练内核）
2. 按 `source_datasets` 顺序循环训练，权重连续继承
3. 每个源域完成后输出中间 checkpoint（可选）
4. 最终输出 consolidated 预训练权重到 `models_pretrain/...`

完成标准：

- 在小样本短轮次下可生成有效 `.pkl` 预训练权重

---

### M3: 目标域微调接入

任务：

1. 在 `model_train.py` 接入可选预训练加载参数通道
2. 若指定预训练权重则加载后训练；否则保持原行为
3. 微调时应用学习率缩放（默认 `0.1`）
4. 与课程学习 stage1/2/3 逻辑兼容

完成标准：

- 使用预训练权重可成功在 `CurveFaultA` 上启动并训练

---

### M4: 评估与对照

任务：

1. 准备 baseline/proposed 模型清单 JSON
2. 通过 `model_test.py --compare-config ...` 统一测试
3. 导出对照 CSV，按 `mse_mean` 主排序
4. 记录关键运行元数据（模型路径、训练轮数、时间）

完成标准：

- 同一测试集下得到 baseline/proposed 的可比较结果

---

### M5: 稳定性与回退

任务：

1. 数据缺失时单源域跳过并记录
2. 权重 shape 不兼容时回退从头训练并告警
3. NaN 触发一次降学习率重试
4. 整体失败时保证原链路命令可直接运行

完成标准：

- 任意异常不会破坏现有 `model_train.py`/`model_test.py` 可用性

## 4. 验收清单

### 功能验收

- [ ] 可执行源域预训练并保存权重
- [ ] 可加载预训练权重执行目标域微调
- [ ] 评估链路可输出 baseline/proposed 对照结果

### 指标验收

- [ ] 主指标 `mse_mean`：proposed 优于 baseline（快速验证批次）
- [ ] 建议 3 个随机种子中至少 2 次优于 baseline

### 工程验收

- [ ] 不破坏现有训练与测试命令
- [ ] 预训练与正式模型目录完全隔离
- [ ] 核心运行步骤在文档中可复现

## 5. 执行顺序（建议）

1. 先完成 M1（参数与路径）并 dry-run
2. 完成 M2（可产出权重）后做一次短跑
3. 接入 M3 并在 `CurveFaultA` 上跑最小微调
4. 完成 M4 形成第一版对照 CSV
5. 最后补 M5 的鲁棒性处理

## 6. 风险与缓解

- 风险：源域迁移收益不明显
  - 缓解：先调整源域顺序与微调学习率缩放
- 风险：训练时间超预算
  - 缓解：先固定小样本+短 epoch 验证方向
- 风险：权重兼容问题
  - 缓解：模型类型固定为 `DDNet70`，先锁定单架构

## 7. 最小运行命令草案
```bash
cd  D:\bishe\ddnet-master
```
```bash
# 1) 源域预训练（示例）
 --source-datasets FlatVelA,CurveVelA --target-dataset CurveFaultA --pretrain-epochs 5
python pretrain_entry.py
# 2) 目标域微调（示例，具体参数入口按实现后命名）
python model_train.py --load-pretrained <models_pretrain/.../xxx.pkl>

# 3) 统一评估对照
python model_test.py --compare-config compare_models.json
```

## 8. 完成定义（DoD）

当以下条件满足时，本计划视为完成：

1. 预训练->微调->评估闭环可运行
2. 结果文件可追踪并可复现
3. 与基线相比的 `mse_mean` 对照结果可被量化展示
