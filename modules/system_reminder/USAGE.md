# 可视化页面使用手册

## 1. 启动可视化界面

### 1.1 环境准备

- 确保已安装 `PyQt5`
- 在项目根目录执行命令

### 1.2 启动命令

推荐使用演示宿主启动：

```bash
python demo/qt_host_demo.py
```

若提示模块导入失败，请确认当前工作目录是仓库根目录：

```bash
cd D:\bishe\ddnet-master
python demo/qt_host_demo.py
```

启动成功后会打开主窗口，中央即为系统可视化页面容器。

---

## 2. 页面入口

- 打开系统首页 `SystemReminderPage`
- 点击 `训练页面` 进入 `TrainingView`
- 点击 `预测对比页面` 进入 `CompareView`

---

## 3. TrainingView（Loss 可视化）

### 2.1 基本使用

1. 在左侧下拉框选择数据集（如 `SEGSalt`、`CurveFaultA`）
2. 点击 `显示Loss页面`
3. 右侧会显示 Loss 曲线图

### 2.2 放大与下载

- 左键点击右侧 Loss 图可打开放大窗口
- 在放大窗口点击 `下载Loss图` 可导出图像（支持 `png/jpg/bmp`）

### 2.3 异常提示

- 若数据文件缺失、为空或读取失败，右侧图表容器会显示错误信息

---

## 4. CompareView（PD/GT 对比）

### 3.1 布局说明

- 右侧为上下两张图（比例一致）：
  - 上：`PD`
  - 下：`GT`

### 3.2 按钮说明

- `Test`：执行与 `PD` 相同操作
- `PD`：按 `train_or_test="test"` 更新 PD 图
- `GT`：按 `train_or_test="train"` 更新 GT 图
- `返回`：返回上一页

### 3.3 放大与下载

- 点击右侧任意一张图可打开放大窗口
- 放大窗口内点击 `下载图像` 可保存当前图（支持 `png/jpg/bmp`）

---

## 5. 数据处理逻辑说明（与测试脚本一致）

`CompareView` 中速度模型处理逻辑对齐 `model_test.py`：

- 若数据集是 `SEGSalt` / `SEGSimulation`：
  - 使用 `single_read_matfile(...)`
  - `max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)`

- 其他数据集：
  - 使用 `single_read_npyfile(...)`
  - `max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)`
  - 再做归一化：
    - `velocity_model = (velocity_model - min) / (max - min)`

- 可视化显示使用：
  - `minV = np.min(min_velocity + velocity_model * (max_velocity - min_velocity))`
  - `maxV = np.max(min_velocity + velocity_model * (max_velocity - min_velocity))`

---

## 6. 常见问题

- 图像不显示：先确认数据目录和样本索引配置正确
- 文件读取报错：检查文件是否存在、是否为空、格式是否正确
- 下载失败：检查保存目录权限
