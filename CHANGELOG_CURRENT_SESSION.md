# DD-Net Current Session Change Log

This document summarizes the changes made so far in the current workspace.

## 1) CPU/CUDA Compatibility Updates

### `model_train.py`
- Added `import os`.
- Added env switch `DDNET_FORCE_CPU` support:
  - Legacy behavior kept in comments.
  - New logic: CUDA is used only when available and not force-disabled.
- `determine_network(...)` now moves model to CPU explicitly when CUDA is not used.
- In training loop, tensor transfer is unified to `.to(model_device)` with non-blocking enabled only for CUDA.
- Kept legacy `.cuda(...)` transfer code as comments.
- Fixed string comparison from `is not ""` to `!= ""`.
- Added one optional debug line comment for `batch_read_npyfile(...)` small-sample loading.

### `model_test.py`
- Added `import os` (currently unused but harmless).
- `lpips` model is moved to the same runtime device as network (`.to(device)`).
- Input tensors for test are moved by `.to(device)` instead of CUDA-only branch.
- Kept legacy `.cuda(...)` code as comments.
- Fixed batch metric loop to use real batch length (`pd_vmodel.shape[0]`) instead of fixed `test_batch_size`.
- Kept legacy fixed-loop behavior as comment.

### `inversionnet_train.py`
- Added `import os`.
- Added `DDNET_FORCE_CPU` support with legacy behavior kept in comments.
- Model placement now supports CPU path (`InvNet.to(device)` when not using CUDA).
- Batch tensors move via `.to(device)`.
- Kept legacy CUDA-only transfer code as comments.
- Fixed string comparison from `is not ""` to `!= ""`.
- Local debug settings changed:
  - `Epochs` changed from `120` to `2`.
  - `TrainSize` changed to `3`.
  - `save_times` changed from `12` to `1`.

### `fcnvmb_train.py`
- Imports updated to include both readers:
  - `batch_read_matfile, batch_read_npyfile`.
- Added `import os` and `DDNET_FORCE_CPU` support.
- Added CPU model placement path (`fcnNet.to(device)`) when CUDA not used.
- Batch tensors move via `.to(device)`.
- Kept legacy CUDA-only transfer code as comments.
- Fixed string comparison from `is not ""` to `!= ""`.
- Added dataset-dependent loading branch:
  - SEG datasets use `.mat` reader.
  - OpenFWI-style datasets use `.npy` reader.
- Fixed forward call mismatch:
  - from `fcnNet(images)`
  - to `fcnNet(images, model_dim)`.
- Local debug settings changed:
  - `Epochs` changed from `100` to `2`.
  - `TrainSize` changed to `2`.
  - `save_times` changed from `2` to `1`.

### `func/utils.py`
- `model_reader(...)` now loads checkpoints with device mapping:
  - Legacy kept as comment: `torch.load(save_src)`
  - Current: `torch.load(save_src, map_location=device)`
- This allows loading GPU-saved checkpoints on CPU environment.

## 2) Loss Function Device Fixes

### `net/DDNet.py`
- `LossDDNet` no longer creates class weights using hardcoded `.cuda()`.
- Replaced with device-agnostic tensor and `F.cross_entropy(..., weight=...)` using runtime output device.

### `net/DDNet70.py`
- Same device-agnostic `LossDDNet` fix as above.
- Also fixed undefined class names in architecture wiring:
  - Replaced `unetUp1/netUp1/unetUp2/netUp2`
  - with existing `unetUp/netUp` classes.

## 3) Runtime Bug Fix

### `net/InversionNet.py`
- Added missing import:
  - `import torch.nn.functional as F`
- Required for existing `F.pad(...)` in forward pass.

## 4) Environment/Execution Actions Performed (No source code change)

- Installed CPU PyTorch in current global Python (user-site):
  - `torch==2.10.0+cpu`, `torchvision==0.25.0+cpu`, `torchaudio==2.10.0+cpu`
- Verified CPU mode:
  - `torch.version.cuda == None`, `torch.cuda.is_available() == False`
- Configured usage of conda executable provided by user:
  - `D:\ProgramData\miniconda3\Scripts\conda.exe`
- Verified and used conda env:
  - `dataset-profiler`
- Installed/verified dependencies in `dataset-profiler`:
  - `scipy`, `lpips`
- Re-ran training scripts and diagnosed runtime errors step-by-step.

## 5) Notes on Current Workspace State

- `git status` indicates additional files changed in workspace that may be unrelated to this session (for example: `README.md`, `param_config.py`, `path_config.py`, data folders, model artifacts, cache folders).
- This document focuses on the technical changes observed and applied during this session.
