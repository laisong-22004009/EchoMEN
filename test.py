import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import gmean
from utils import loadvideo, get_mean_and_std
from model import r2plus1d_18_stage1
from datasets import Echo_One

import numpy as np
from scipy.stats import gmean

import numpy as np
from scipy.stats import gmean
from sklearn.metrics import r2_score

def shot_metrics_r2(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    # 初始化指标存储
    shot_dict = defaultdict(lambda: defaultdict(float))
    count_dict = defaultdict(int)

    # 定义EF范围
    ef_range_labels = {
        # Existing ranges...
    }

    # 初始化每个范围的指标
    for range_label in ef_range_labels.keys():
        shot_dict[range_label] = {
            'mse': 0,
            'l1': 0,
            'gmean': [],
            'r2': []  # 收集每个范围内的预测值和实际值，以便计算R2分数
        }
        count_dict[range_label] = 0

    # 计算每个范围的指标
    for i, label in enumerate(labels):
        for range_label, (low, high) in ef_range_labels.items():
            if low <= label < high:
                mse = (preds[i] - label) ** 2
                l1 = np.abs(preds[i] - label)

                shot_dict[range_label]['mse'] += mse
                shot_dict[range_label]['l1'] += l1
                shot_dict[range_label]['gmean'].append(l1)
                if 'preds' not in shot_dict[range_label]:
                    shot_dict[range_label]['preds'] = []
                    shot_dict[range_label]['true'] = []
                shot_dict[range_label]['preds'].append(preds[i])
                shot_dict[range_label]['true'].append(label)
                count_dict[range_label] += 1

    # 计算平均指标和R2分数
    for range_label in ef_range_labels.keys():
        shot_dict[range_label]['mse'] /= count_dict[range_label] if count_dict[range_label] != 0 else 1
        shot_dict[range_label]['l1'] /= count_dict[range_label] if count_dict[range_label] != 0 else 1
        shot_dict[range_label]['gmean'] = gmean(shot_dict[range_label]['gmean'], axis=None).astype(float)
        # 使用sklearn.metrics.r2_score计算R2分数
        if count_dict[range_label] > 1:  # R2分数至少需要两个数据点
            shot_dict[range_label]['r2'] = r2_score(shot_dict[range_label]['true'], shot_dict[range_label]['preds'])
        else:
            shot_dict[range_label]['r2'] = 'N/A'

    return shot_dict



def shot_metrics(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    # 初始化指标存储
    shot_dict = defaultdict(lambda: defaultdict(float))
    count_dict = defaultdict(int)

    # 定义EF范围
    ef_range_labels = {
        # "1": (0, 10),
        # "2": (10, 20),
        # "3": (20, 30),
        # "4": (30, 40),
        # "5": (40, 50),
        # "6": (50, 60),
        # "7": (60, 70),
        # "8": (70, 80),
        # "9": (80, 90),
        # "10": (90, 100),
        # "AHyperdynamic": (70, 100),
        # "Normal EF": (50, 70),
        # "Mild dysfunction EF": (40, 50),
        # "Moderate dysfunction EF": (30, 40),
        # "Severe dysfunction EF": (0, 30),
        "Ep EF": (50, 100),
        "Emr EF": (40, 50),
        "Er EF": (0, 40),
        "all": (0, 100),
    }

    # 初始化每个范围的指标
    for range_label in ef_range_labels.keys():
        shot_dict[range_label] = {
            'mse': 0,
            'l1': 0,
            'gmean': []
        }
        count_dict[range_label] = 0

    # 计算每个范围的指标
    for i, label in enumerate(labels):
        for range_label, (low, high) in ef_range_labels.items():
            if low <= label < high:
                mse = (preds[i] - label) ** 2
                l1 = np.abs(preds[i] - label)

                shot_dict[range_label]['mse'] += mse
                shot_dict[range_label]['l1'] += l1
                shot_dict[range_label]['gmean'].append(l1)
                count_dict[range_label] += 1

    # 计算平均指标
    for range_label in ef_range_labels.keys():
        shot_dict[range_label]['mse'] /= count_dict[range_label] if count_dict[range_label] != 0 else 1
        shot_dict[range_label]['l1'] /= count_dict[range_label] if count_dict[range_label] != 0 else 1
        shot_dict[range_label]['gmean'] = gmean(shot_dict[range_label]['gmean'], axis=None).astype(float)

    return shot_dict


def test_epoch_all(model, fnames, device, mean, std, length, period, run_dir, block_size=32):
    DATA_DIR = "/home/****/EchoNet/a4c-video-dir/"
    model.eval()
    yhat = []
    filenames = []

    # 使用tqdm进度条显示测试进度
    for file in tqdm(fnames, desc="Testing"):
        video_path = os.path.join(DATA_DIR, "Videos", file)
        video = loadvideo(video_path).astype(np.float32)

        # 标准化处理
        video = (video - mean.reshape(3, 1, 1, 1)) / std.reshape(3, 1, 1, 1)

        # 调整视频长度
        c, f, h, w = video.shape
        length = min(length, f // period)
        if f < length * period:
            video = np.concatenate((video, np.zeros((c, length * period - f, h, w), video.dtype)), axis=1)

        # 分块处理和预测
        starts = np.arange(0, f - (length - 1) * period, block_size * period)
        video_preds = []
        for start in starts:
            end = min(start + block_size * period, f)
            indices = np.arange(start, end, period)
            indices = np.clip(indices, 0, f-1)  # 确保索引不会越界
            if indices.size == 0:
                continue  # 如果没有足够的帧，跳过这个块
            vid_samp = video[:, indices, :, :]
            X1 = torch.tensor(vid_samp[np.newaxis, ...]).to(device)  # 增加批次维度
            with torch.no_grad():
                output = model(X1)
            video_preds.append(output[0].cpu().numpy())
        
        # 计算当前视频的平均预测值
        if video_preds:
            mean_pred = np.mean(video_preds)
        else:
            mean_pred = np.nan  # 如果视频没有产生任何预测，使用NaN
        yhat.append(mean_pred)
        filenames.append(file)

    # 保存预测结果
    predict_out_path = os.path.join(run_dir, "predictions.csv")
    predictions = pd.DataFrame({"FileName": filenames, "PredictedEF": yhat})
    label_data_path = os.path.join(DATA_DIR, "FileList.csv")
    label_data = pd.read_csv(label_data_path)
    ###############################
    label_data_select = label_data[['FileName', 'EF']]
    # 把label_data_select中的FileName全部加上.avi后缀
    label_data_select['FileName'] = label_data_select['FileName'].apply(lambda x: x + ".avi")
    ###############################
    with_predict = predictions.merge(label_data_select, left_on='FileName', right_on='FileName')

    with_predict.to_csv(predict_out_path, index=False)

    # 使用shot_metrics分析结果
    shot_metrics_dict = shot_metrics(with_predict['PredictedEF'].to_numpy(), with_predict['EF'].to_numpy())

    # 将shot_metrics结果写入文件
    metrics_out_path = os.path.join(run_dir, "metrics_analysis.csv")
    with open(metrics_out_path, 'w') as f:
        for category, metrics in shot_metrics_dict.items():
            gmean_str = "NaN" if np.isnan(metrics['gmean']) else f"{metrics['gmean']}"
            f.write(f"{category}, MSE: {metrics['mse']}, L1: {metrics['l1']}, G-Mean: {gmean_str}\n")

    return with_predict['PredictedEF'].to_numpy(), with_predict['EF'].to_numpy()

def test_epoch_all_print(model, fnames, device, mean, std, length, period, block_size=32):
    DATA_DIR = "/home/****/EchoNet/a4c-video-dir/"
    model.eval()
    yhat = []
    filenames = []

    # 使用tqdm进度条显示测试进度
    for file in tqdm(fnames, desc="Testing"):
        video_path = os.path.join(DATA_DIR, "Videos", file)
        video = loadvideo(video_path).astype(np.float32)

        # 标准化处理
        video = (video - mean.reshape(3, 1, 1, 1)) / std.reshape(3, 1, 1, 1)

        # 调整视频长度
        c, f, h, w = video.shape
        length = min(length, f // period)
        if f < length * period:
            video = np.concatenate((video, np.zeros((c, length * period - f, h, w), video.dtype)), axis=1)

        # 分块处理和预测
        starts = np.arange(0, f - (length - 1) * period, block_size * period)
        video_preds = []
        for start in starts:
            end = min(start + block_size * period, f)
            indices = np.arange(start, end, period)
            indices = np.clip(indices, 0, f-1)  # 确保索引不会越界
            if indices.size == 0:
                continue  # 如果没有足够的帧，跳过这个块
            vid_samp = video[:, indices, :, :]
            X1 = torch.tensor(vid_samp[np.newaxis, ...]).to(device)  # 增加批次维度
            with torch.no_grad():
                output = model(X1)
            video_preds.append(output[0].cpu().numpy())
        
        # 计算当前视频的平均预测值
        if video_preds:
            mean_pred = np.mean(video_preds)
        else:
            mean_pred = np.nan  # 如果视频没有产生任何预测，使用NaN
        yhat.append(mean_pred)
        filenames.append(file)

    # 保存预测结果
    predictions = pd.DataFrame({"FileName": filenames, "PredictedEF": yhat})
    label_data_path = os.path.join(DATA_DIR, "FileList.csv")
    label_data = pd.read_csv(label_data_path)
    ###############################
    label_data_select = label_data[['FileName', 'EF']]
    # 把label_data_select中的FileName全部加上.avi后缀
    label_data_select['FileName'] = label_data_select['FileName'].apply(lambda x: x + ".avi")
    ###############################
    with_predict = predictions.merge(label_data_select, left_on='FileName', right_on='FileName')

    # 使用shot_metrics分析结果
    shot_metrics_dict = shot_metrics(with_predict['PredictedEF'].to_numpy(), with_predict['EF'].to_numpy())

 
    for category, metrics in shot_metrics_dict.items():
        print(f" * {category.capitalize()}: MSE {metrics['mse']:.3f}\t"
                f"L1 {metrics['l1']:.3f}\tG-Mean {metrics['gmean']:.3f}")

    return with_predict['PredictedEF'].to_numpy(), with_predict['EF'].to_numpy()


if __name__ == "__main__":
    # 加载已经训练好的模型
    device = torch.device("cuda")
    model = r2plus1d_18_stage1()
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6
    if device.type == "cuda":
        model = model.to(device)

    weights_path = "/home/****/EchoNet/AdaCon/results/Ada_noctr/best.pt"
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['state_dict'])

    # 准备数据集
    frames = 32
    period = 2
    mean_dataset = Echo_One(split="train")
    print("len of mean dataset", len(mean_dataset))
    mean, std = get_mean_and_std(mean_dataset)
    print("mean std", mean, std)
    kwargs = {  "mean": mean,
                "std": std,
                "length": frames,
                "period": period,
                }

    test_dataset = Echo_One(split="test", **kwargs)
    # 首先获取全部测试集文件名
    test_fnames = test_dataset.fnames  # 假设这样可以获取到所有文件名

    # # 然后随机选择十分之一的文件名
    # sampled_fnames = np.random.choice(test_fnames, size=len(test_fnames) // 10, replace=False)


    run_dir = "/home/****/EchoNet/AdaCon/mlp_results/Ada_No"
    test_epoch_all(model, test_fnames, device, mean, std, length=frames, period=period, run_dir=run_dir)