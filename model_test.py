# -*- coding: utf-8 -*-
import os

# Windows 下 OpenMP 运行时冲突绕过方案。
# 用于避免出现 "libiomp5md.dll already initialized" 之类的崩溃，
# 通常由不同依赖重复加载 OpenMP 运行时导致。
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from path_config import *
from func.utils import run_mse, run_mae, run_lpips, run_uqi, pain_seg_seismic_data, pain_seg_velocity_model,\
    pain_openfwi_velocity_model, pain_openfwi_seismic_data
from func.datasets_reader import batch_read_matfile, batch_read_npyfile, single_read_matfile, single_read_npyfile
from model_train import determine_network
from func.device_selector import get_runtime_device

import time
import json
import csv
import lpips
import numpy as np
import torch
import torch.utils.data as data_utils

import matplotlib
matplotlib.use('TkAgg')

VALID_MODEL_TYPES = {"DDNet", "DDNet70", "InversionNet", "FCNVMB", "SDNet", "SDNet70"}


def _model_tag(model_path):
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    return base_name.replace(' ', '_')


def _normalize_path(path):
    return path.replace("\\", "/") if path else path


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def load_compare_config(config_path="compare_models.json"):
    if not os.path.exists(config_path):
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("compare_models.json must be a JSON object")

    default_select_id = 1615 if dataset_name in ["SEGSalt", "SEGSimulation"] else [1, 2]
    models = raw.get("models", [])
    if not isinstance(models, list) or len(models) == 0:
        raise ValueError("compare_models.json must include a non-empty 'models' list")

    return {
        "enabled": _to_bool(raw.get("enabled", True)),
        "run_single": _to_bool(raw.get("run_single", True)),
        "run_batch": _to_bool(raw.get("run_batch", True)),
        "single_select_id": raw.get("single_select_id", default_select_id),
        "save_preview": _to_bool(raw.get("save_preview", True)),
        "show_preview": _to_bool(raw.get("show_preview", False)),
        "sort_by": str(raw.get("sort_by", "mae_mean")),
        "models": models,
    }


def _sort_compare_rows(rows, sort_by):
    if not rows:
        return []

    descending_metrics = {"uqi_mean"}
    sortable_keys = {"mse_mean", "mae_mean", "uqi_mean", "lpips_mean", "elapsed_seconds", "per_sample_seconds"}
    key_name = sort_by if sort_by in sortable_keys else "mae_mean"

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    non_ok_rows = [r for r in rows if r.get("status") != "ok"]

    if key_name in descending_metrics:
        ok_rows.sort(key=lambda r: r.get(key_name, float("-inf")), reverse=True)
    else:
        ok_rows.sort(key=lambda r: r.get(key_name, float("inf")))

    return ok_rows + non_ok_rows


def _save_compare_table(rows, mode_tag, sort_by):
    if not rows:
        return None

    ordered_rows = _sort_compare_rows(rows, sort_by)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, "[Compare{}]{}_{}.csv".format(mode_tag, dataset_name, timestamp))
    fields = [
        "rank",
        "status",
        "alias",
        "model_type",
        "model_path",
        "mse_mean",
        "mae_mean",
        "uqi_mean",
        "lpips_mean",
        "elapsed_seconds",
        "per_sample_seconds",
        "error",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        rank = 1
        for row in ordered_rows:
            export_row = dict(row)
            if row.get("status") == "ok":
                export_row["rank"] = rank
                rank += 1
            else:
                export_row["rank"] = ""
            writer.writerow(export_row)

    print("{} compare table saved: {}".format(mode_tag, output_path))
    return output_path


def run_multi_model_compare(config):
    run_single = config.get("run_single", True)
    run_batch = config.get("run_batch", True)
    if not run_single and not run_batch:
        raise ValueError("At least one of run_single or run_batch must be true")

    print("=" * 50)
    print("自动多模型对比")
    print("Dataset: {}".format(dataset_name))
    print("Run single: {} | Run batch: {}".format(run_single, run_batch))
    print("=" * 50)

    single_rows = []
    batch_rows = []

    for idx, model in enumerate(config["models"]):
        alias = str(model.get("alias", "model_{}".format(idx + 1)))
        model_type = str(model.get("model_type", "")).strip()
        model_path = str(model.get("model_path", "")).strip()
        row_base = {
            "status": "ok",
            "alias": alias,
            "model_type": model_type,
            "model_path": model_path,
            "mse_mean": "",
            "mae_mean": "",
            "uqi_mean": "",
            "lpips_mean": "",
            "elapsed_seconds": "",
            "per_sample_seconds": "",
            "error": "",
        }

        print("\n[Model {}] {} ({})".format(idx + 1, alias, model_type))

        if model_type not in VALID_MODEL_TYPES:
            error_msg = "invalid_model_type"
            print("Skip: {} is not supported.".format(model_type))
            if run_single:
                row = dict(row_base)
                row.update({"status": "invalid_model_type", "error": error_msg})
                single_rows.append(row)
            if run_batch:
                row = dict(row_base)
                row.update({"status": "invalid_model_type", "error": error_msg})
                batch_rows.append(row)
            continue

        if not os.path.exists(model_path):
            error_msg = "model_file_not_found"
            print("Skip: model file not found -> {}".format(model_path))
            if run_single:
                row = dict(row_base)
                row.update({"status": "missing", "error": error_msg})
                single_rows.append(row)
            if run_batch:
                row = dict(row_base)
                row.update({"status": "missing", "error": error_msg})
                batch_rows.append(row)
            continue

        if run_single:
            try:
                single_result = single_test(
                    model_path,
                    select_id=config["single_select_id"],
                    model_type=model_type,
                    save_preview=config["save_preview"],
                    show_preview=config["show_preview"],
                )
                row = dict(row_base)
                row.update(single_result)
                row["per_sample_seconds"] = row.get("elapsed_seconds", "")
                single_rows.append(row)
            except Exception as e:
                row = dict(row_base)
                row.update({"status": "failed", "error": str(e)})
                single_rows.append(row)
                print("Single test failed: {}".format(e))

        if run_batch:
            try:
                batch_result = batch_test(model_path, model_type=model_type)
                row = dict(row_base)
                row.update(batch_result)
                batch_rows.append(row)
            except Exception as e:
                row = dict(row_base)
                row.update({"status": "failed", "error": str(e)})
                batch_rows.append(row)
                print("Batch test failed: {}".format(e))

    if run_single:
        _save_compare_table(single_rows, "Single", config.get("sort_by", "mae_mean"))
    if run_batch:
        _save_compare_table(batch_rows, "Batch", config.get("sort_by", "mae_mean"))


def _save_batch_metrics(model_path, model_type, mse, mae, uqi, lpips_score, elapsed_seconds):
    tag = _model_tag(model_path)
    save_prefix = "[Metric]{}_{}_{}".format(dataset_name, model_type, tag)

    with open(os.path.join(results_dir, save_prefix + "_summary.txt"), "w", encoding="utf-8") as f:
        f.write("dataset: {}\n".format(dataset_name))
        f.write("model_type: {}\n".format(model_type))
        f.write("model_path: {}\n".format(model_path))
        f.write("sample_count: {}\n".format(len(mse)))
        f.write("mse_mean: {:.8f}\n".format(np.mean(mse)))
        f.write("mae_mean: {:.8f}\n".format(np.mean(mae)))
        f.write("uqi_mean: {:.8f}\n".format(np.mean(uqi)))
        f.write("lpips_mean: {:.8f}\n".format(np.mean(lpips_score)))
        f.write("elapsed_seconds: {:.6f}\n".format(elapsed_seconds))
    print("Summary saved: {}".format(os.path.join(results_dir, save_prefix + "_summary.txt")))


def _save_single_metrics(model_path, model_type, select_id, mse, mae, uqi, lpips_score, elapsed_seconds):
    tag = _model_tag(model_path)
    save_prefix = "[SingleMetric]{}_{}_{}".format(dataset_name, model_type, tag)

    with open(os.path.join(results_dir, save_prefix + "_summary.txt"), "w", encoding="utf-8") as f:
        f.write("dataset: {}\n".format(dataset_name))
        f.write("model_type: {}\n".format(model_type))
        f.write("model_path: {}\n".format(model_path))
        f.write("select_id: {}\n".format(select_id))
        f.write("mse: {:.8f}\n".format(mse))
        f.write("mae: {:.8f}\n".format(mae))
        f.write("uqi: {:.8f}\n".format(uqi))
        f.write("lpips: {:.8f}\n".format(lpips_score))
        f.write("elapsed_seconds: {:.6f}\n".format(elapsed_seconds))
    print("Single-sample summary saved: {}".format(os.path.join(results_dir, save_prefix + "_summary.txt")))


def load_dataset():
    '''
    根据 "param_config" 中的参数加载测试数据

    :return:    A triplet: datasets loader, seismic gathers and velocity models
    '''

    print("---------------------------------")
    print("· Loading the datasets...")
    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        data_set, label_sets = batch_read_matfile(data_dir, 1 if dataset_name == 'SEGSalt'
                                                              else 1601, test_size, "test")
    else:
        data_set, label_sets = batch_read_npyfile(data_dir, 1, test_size // 500, "test")
        for i in range(data_set.shape[0]):
            vm = label_sets[0][i][0]
            max_velocity, min_velocity = np.max(vm), np.min(vm)
            label_sets[0][i][0] = (vm - min_velocity) / (max_velocity - min_velocity)

    print("· Number of seismic gathers included in the testing set: {}.".format(test_size))
    print("· Dimensions of seismic data: ({},{},{},{}).".format(test_size, inchannels, data_dim[0], data_dim[1]))
    print("· Dimensions of velocity model: ({},{},{},{}).".format(test_size, classes, model_dim[0], model_dim[1]))
    print("---------------------------------")

    seis_and_vm = data_utils.TensorDataset(torch.from_numpy(data_set).float(),
                                           torch.from_numpy(label_sets[0]).float())
    seis_and_vm_loader = data_utils.DataLoader(seis_and_vm, batch_size=test_batch_size, shuffle=True)

    return seis_and_vm_loader, data_set, label_sets

def batch_test(model_path, model_type = "DDNet"):
    '''
    多样本批量测试

    :param model_path:              模型路径
    :param model_type:              主模型类型。
                                    可用关键字：
                                    [DDNet70 | DDNet | InversionNet | FCNVMB | SDNet70 | SDNet]
    :return:
    '''

    loader, seismic_gathers, velocity_models = load_dataset()

    device, _, resolved_mode = get_runtime_device(device_mode)
    print("[Device] mode={} resolved={} cuda_available={}".format(resolved_mode, device.type, torch.cuda.is_available()))
    print("Loading test model:{}".format(model_path))
    model_net, device, optimizer = determine_network(model_path, model_type=model_type)

    mse_record = np.zeros((1, test_size), dtype=float)
    mae_record = np.zeros((1, test_size), dtype=float)
    uqi_record = np.zeros((1, test_size), dtype=float)
    lpips_record = np.zeros((1, test_size), dtype=float)

    counter = 0

    # 旧版本写法：lpips_object = lpips.LPIPS(net='alex', version="0.1")
    lpips_object = lpips.LPIPS(net='alex', version="0.1").to(device)

    cur_node_time = time.time()
    for i, (seis_image, gt_vmodel) in enumerate(loader):

        # 旧版本写法：
        # if torch.cuda.is_available():
        #     seis_image = seis_image.cuda(non_blocking=True)
        #     gt_vmodel = gt_vmodel.cuda(non_blocking=True)
        seis_image = seis_image.to(device, non_blocking=device.type == "cuda")
        gt_vmodel = gt_vmodel.to(device, non_blocking=device.type == "cuda")

        # 预测
        model_net.eval()
        if model_type in ["DDNet", "DDNet70"]:
            [outputs, _] = model_net(seis_image, model_dim)
        elif model_type in ["SDNet", "SDNet70"]:
            outputs = model_net(seis_image, model_dim)
        elif model_type == "InversionNet":
            outputs = model_net(seis_image)
        elif model_type == "FCNVMB":
            outputs = model_net(seis_image, model_dim)
        else:
            print('The "model_type" parameter selected in the batch_test(...) '
                  'is the undefined network model keyword! Please check!')
            exit(0)

        # 将标签与预测结果都转为 numpy
        pd_vmodel = outputs.cpu().detach().numpy()
        pd_vmodel = np.where(pd_vmodel > 0.0, pd_vmodel, 0.0)   # 裁剪掉非物理负值
        gt_vmodel = gt_vmodel.cpu().detach().numpy()

        # 计算当前 batch 的 MSE、MAE、UQI 和 LPIPS
        # 旧版本写法：for k in range(test_batch_size):
        cur_batch = pd_vmodel.shape[0]
        for k in range(cur_batch):

            pd_vmodel_of_k = pd_vmodel[k, 0, :, :]
            gt_vmodel_of_k = gt_vmodel[k, 0, :, :]

            mse_record[0, counter]   = run_mse(pd_vmodel_of_k, gt_vmodel_of_k)
            mae_record[0, counter]   = run_mae(pd_vmodel_of_k, gt_vmodel_of_k)
            uqi_record[0, counter]   = run_uqi(gt_vmodel_of_k, pd_vmodel_of_k)
            lpips_record[0, counter] = run_lpips(gt_vmodel_of_k, pd_vmodel_of_k, lpips_object)

            print('The %d testing MSE: %.4f\tMAE: %.4f\tUQI: %.4f\tLPIPS: %.4f' %
                  (counter, mse_record[0, counter], mae_record[0, counter],
                   uqi_record[0, counter], lpips_record[0, counter]))
            counter = counter + 1
    time_elapsed = time.time() - cur_node_time

    valid_mse = mse_record[0, :counter]
    valid_mae = mae_record[0, :counter]
    valid_uqi = uqi_record[0, :counter]
    valid_lpips = lpips_record[0, :counter]
    mse_mean = float(np.mean(valid_mse))
    mae_mean = float(np.mean(valid_mae))
    uqi_mean = float(np.mean(valid_uqi))
    lpips_mean = float(np.mean(valid_lpips))
    per_sample_seconds = float(time_elapsed / counter) if counter > 0 else 0.0

    print("The average of MSE: {:.6f}".format(mse_mean))
    print("The average of MAE: {:.6f}".format(mae_mean))
    print("The average of UQI: {:.6f}".format(uqi_mean))
    print("The average of LIPIS: {:.6f}".format(lpips_mean))
    print("-----------------")
    print("Time-consuming testing of batch samples: {:.6f}".format(time_elapsed))
    print("Average test-consuming per sample: {:.6f}".format(per_sample_seconds))

    _save_batch_metrics(
        model_path,
        model_type,
        valid_mse,
        valid_mae,
        valid_uqi,
        valid_lpips,
        time_elapsed,
    )

    return {
        "status": "ok",
        "mse_mean": mse_mean,
        "mae_mean": mae_mean,
        "uqi_mean": uqi_mean,
        "lpips_mean": lpips_mean,
        "elapsed_seconds": float(time_elapsed),
        "per_sample_seconds": per_sample_seconds,
    }


def single_test(model_path, select_id, train_or_test = "test", model_type = "DDNet", save_preview = 1, show_preview = 1):
    '''
    单样本测试

    :param model_path:              模型路径
    :param select_id:               选中样本 ID。
                                    OpenFWI 一般为二元组，如 [11, 100]；
                                    SEG 数据一般为单值，如 56。
    :param train_or_test:           数据属于训练集还是测试集
    :param model_type:              主模型类型。
                                    可用关键字：
                                    [DDNet70 | DDNet | InversionNet | FCNVMB | SDNet70 | SDNet]
    :return:
    '''

    device, _, resolved_mode = get_runtime_device(device_mode)
    print("[Device] mode={} resolved={} cuda_available={}".format(resolved_mode, device.type, torch.cuda.is_available()))
    print("Loading test model:{}".format(model_path))
    model_net, device, optimizer = determine_network(model_path, model_type=model_type)

    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        seismic_data, velocity_model, _ = single_read_matfile(data_dir, data_dim, model_dim, select_id, train_or_test = train_or_test)
        max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
    else:
        seismic_data, velocity_model, _ = single_read_npyfile(data_dir, select_id, train_or_test = train_or_test)
        max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
        velocity_model = (velocity_model - np.min(velocity_model)) / (np.max(velocity_model) - np.min(velocity_model))

    # 旧版本写法：lpips_object = lpips.LPIPS(net='alex', version="0.1")
    lpips_object = lpips.LPIPS(net='alex', version="0.1").to(device)


    # 将 numpy 转为 tensor 并加载到设备
    seismic_data_tensor = torch.from_numpy(np.array([seismic_data])).float()
    # 旧版本写法：
    # if torch.cuda.is_available():
    #     seismic_data_tensor = seismic_data_tensor.cuda(non_blocking=True)
    seismic_data_tensor = seismic_data_tensor.to(device, non_blocking=device.type == "cuda")

    # 预测
    model_net.eval()
    cur_node_time = time.time()
    if model_type in ["DDNet", "DDNet70"]:
        [predicted_vmod_tensor, _] = model_net(seismic_data_tensor, model_dim)
    elif model_type in ["SDNet", "SDNet70"]:
        predicted_vmod_tensor = model_net(seismic_data_tensor, model_dim)
    elif model_type == "InversionNet":
        predicted_vmod_tensor = model_net(seismic_data_tensor)
    elif model_type == "FCNVMB":
        predicted_vmod_tensor = model_net(seismic_data_tensor, model_dim)
    else:
        print('The "model_type" parameter selected in the single_test(...) '
              'is the undefined network model keyword! Please check!')
        exit(0)
    time_elapsed = time.time() - cur_node_time

    predicted_vmod = predicted_vmod_tensor.cpu().detach().numpy()[0][0]     # (1, 1, X, X)
    predicted_vmod = np.where(predicted_vmod > 0.0, predicted_vmod, 0.0)    # 裁剪掉非物理负值

    mse   = run_mse(predicted_vmod, velocity_model)
    mae   = run_mae(predicted_vmod, velocity_model)
    uqi   = run_uqi(velocity_model, predicted_vmod)
    lpi = run_lpips(velocity_model, predicted_vmod, lpips_object)

    print('MSE: %.6f\nMAE: %.6f\nUQI: %.6f\nLPIPS: %.6f' % (mse, mae, uqi, lpi))
    print("-----------------")
    print("Time-consuming testing of a sample: {:.6f}".format(time_elapsed))

    _save_single_metrics(model_path, model_type, select_id, mse, mae, uqi, lpi, time_elapsed)

    '''
    # 显示和/或保存预览图
    save_preview = bool(save_preview)
    show_preview = bool(show_preview)

    if not save_preview and not show_preview:
        print("Preview disabled: save_preview=0, show_preview=0")
        return {
            "status": "ok",
            "mse_mean": float(mse),
            "mae_mean": float(mae),
            "uqi_mean": float(uqi),
            "lpips_mean": float(lpi),
            "elapsed_seconds": float(time_elapsed),
            "per_sample_seconds": float(time_elapsed),
        }

    preview_dir = _normalize_path(os.path.join(results_dir, "previews"))
    if save_preview:
        os.makedirs(preview_dir, exist_ok=True)

    safe_select_id = str(select_id).replace(' ', '').replace('[', '').replace(']', '').replace(',', '-')
    preview_prefix = "{}_{}_{}_{}".format(dataset_name, model_type, _model_tag(model_path), safe_select_id)
    seismic_path = _normalize_path(os.path.join(preview_dir, preview_prefix + "_seismic.png")) if save_preview else None
    gt_path = _normalize_path(os.path.join(preview_dir, preview_prefix + "_gt.png")) if save_preview else None
    pred_path = _normalize_path(os.path.join(preview_dir, preview_prefix + "_pred.png")) if save_preview else None


    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        pain_seg_seismic_data(
            seismic_data[15],
            save_path=seismic_path,
            show=show_preview,
        )
        pain_seg_velocity_model(
            velocity_model,
            min_velocity,
            max_velocity,
            save_path=gt_path,
            show=show_preview,
        )
        pain_seg_velocity_model(
            predicted_vmod,
            min_velocity,
            max_velocity,
            save_path=pred_path,
            show=show_preview,
        )
    else:
        pain_openfwi_seismic_data(
            seismic_data[2],
            save_path=seismic_path,
            show=show_preview,
        )
        minV = np.min(min_velocity + velocity_model * (max_velocity - min_velocity))
        maxV = np.max(min_velocity + velocity_model * (max_velocity - min_velocity))
        pain_openfwi_velocity_model(
            min_velocity + velocity_model * (max_velocity - min_velocity),
            minV,
            maxV,
            save_path=gt_path,
            show=show_preview,
        )
        pain_openfwi_velocity_model(
            min_velocity + predicted_vmod * (max_velocity - min_velocity),
            minV,
            maxV,
            save_path=pred_path,
            show=show_preview,
        )

    if save_preview:
        print("Preview figures saved to: {}".format(preview_dir))
    if show_preview:
        print("Preview windows shown.")

    return {
        "status": "ok",
        "mse_mean": float(mse),
        "mae_mean": float(mae),
        "uqi_mean": float(uqi),
        "lpips_mean": float(lpi),
        "elapsed_seconds": float(time_elapsed),
        "per_sample_seconds": float(time_elapsed),
    }
'''
if __name__ == "__main__":
    # 测试手动参数（不通过命令行传入）
    TEST_MANUAL_CONFIG = {
        "compare_config": "compare_models.pretrain.json",
        "enable_lpips": False,
    }

    compare_config = None
    try:
        compare_config = load_compare_config(TEST_MANUAL_CONFIG["compare_config"])
    except Exception as e:
        print("Failed to load {}: {}".format(TEST_MANUAL_CONFIG["compare_config"], e))

    if compare_config is not None and compare_config.get("enabled", True):
        run_multi_model_compare(compare_config)
        exit(0)

    batch_of_single =2
    # 可选模型类型：|DDNet|DDNet70|InversionNet|FCNVMB|SDNet|SDNet70|
    model_type = "DDNet70"
    save_preview = 1
    show_preview = 1

    if batch_of_single == 1:
        # 批量测试
        batch_test("models\CurveFaultAModel\CurveFaultA_CLStage1_TrSize3_AllEpo10_CurEpo5.pkl", model_type="DDNet70")
    else:
        # 单样本测试
        if dataset_name in ["SEGSalt", "SEGSimulation"]:
            # 1~10      :SEGSalt
            # 1601~1700 :SEGSimulation
            select_id = 1615
        else:
            # [1~2, 0~499]
            select_id = [1, 2]
        single_test(
            "models\CurveFaultAModel\CurveFaultA_CLStage1_TrSize3_AllEpo10_CurEpo5.pkl",
            select_id=select_id,
            model_type=model_type,
            save_preview=save_preview,
            show_preview=show_preview,
        )

jie