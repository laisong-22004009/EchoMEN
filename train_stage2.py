import argparse
import torch
from model import r2plus1d_18_stage1
from datasets import Echo_One, Echo_One_Reweight
from utils import get_mean_and_std
import torch.nn as nn
import os
import numpy as np
from collections import defaultdict
from scipy.stats import gmean
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from test import shot_metrics_r2, shot_metrics
from utils import loadvideo, get_mean_and_std
import pandas as pd

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

def test_epoch_all_moe(model, fnames, device, mean, std, length, period, run_dir, epoch, block_size=32):
    model.eval()
    DATA_DIR = "/home/****/EchoNet/a4c-video-dir/"
    predictions = {"FileName": fnames}
    ef_keys = ['ef_none', 'ef_inverse', 'ef_sqrt_inv', 'final_ef']
    for key in ef_keys:
        predictions[key] = []
    predictions["weights"] = []

    for file in tqdm(fnames, desc=f"Testing Epoch {epoch}"):
        video_path = os.path.join(DATA_DIR, "Videos", file)
        video = loadvideo(video_path).astype(np.float32)

        video = (video - mean.reshape(3, 1, 1, 1)) / std.reshape(3, 1, 1, 1)

        c, f, h, w = video.shape
        length = min(length, f // period)
        if f < length * period:
            video = np.concatenate((video, np.zeros((c, length * period - f, h, w), video.dtype)), axis=1)
        
        starts = np.arange(0, f - (length - 1) * period, block_size * period)
        video_preds = []
        for start in starts:
            end = min(start + block_size * period, f)
            indices = np.arange(start, end, period)
            indices = np.clip(indices, 0, f-1)
            if indices.size == 0:
                continue
            vid_samp = video[:, indices, :, :]
            X1 = torch.tensor(vid_samp[np.newaxis, ...]).to(device)
            with torch.no_grad():
                ef_none, ef_inverse, ef_sqrt_inv, final_ef, weights = model(X1)
            video_preds.append([ef_none.mean().item(), ef_inverse.mean().item(), ef_sqrt_inv.mean().item(), final_ef.mean().item(), weights.mean(dim=0).cpu().numpy()])

        # 将预测结果和权重添加到相应的列表中
        for i, key in enumerate(ef_keys):
            predictions[key].append(np.mean([pred[i] for pred in video_preds]))
        predictions['weights'].append(np.mean([pred[4] for pred in video_preds], axis=0))

    # 将预测结果转换为DataFrame
    predictions_df = pd.DataFrame(predictions)
    predict_out_path = os.path.join(run_dir, f"integrated_model_predictions_epoch_{epoch}.csv")
    predictions_df.to_csv(predict_out_path, index=False)

    label_data_path = os.path.join(DATA_DIR, "FileList.csv")
    label_data = pd.read_csv(label_data_path)
    label_data_select = label_data[['FileName', 'EF']]
    # 把label_data_select中的FileName全部加上.avi后缀
    label_data_select['FileName'] = label_data_select['FileName'].apply(lambda x: x + ".avi")

    for key in ef_keys:
        with_predict = predictions_df[['FileName', key]].merge(label_data_select, left_on='FileName', right_on='FileName')
        with_predict.rename(columns={key: 'PredictedEF'}, inplace=True)
        shot_metrics_dict = shot_metrics(with_predict['PredictedEF'].to_numpy(), with_predict['EF'].to_numpy())
        
        metrics_out_path = os.path.join(run_dir, f"{key}_metrics_analysis_epoch_{epoch}.csv")
        with open(metrics_out_path, 'w') as f:
            for category, metrics in shot_metrics_dict.items():
                gmean_str = "NaN" if np.isnan(metrics['gmean']) else f"{metrics['gmean']}"
                f.write(f"{category}, MSE: {metrics['mse']}, L1: {metrics['l1']}, G-Mean: {gmean_str}\n")

    return predictions_df


class IntegratedModelWithSelectiveTraining(nn.Module):
    def __init__(self, pretrained_model):
        super(IntegratedModelWithSelectiveTraining, self).__init__()
        # 使用预训练模型作为特征提取器
        self.feature_extractor = pretrained_model
        
        # 确保预训练模型的参数在训练过程中不会改变
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # 获取特征提取层输出特征的数量
        num_features = self.feature_extractor.fc.in_features
        
        # 为每种reweight策略定义一个全连接层
        self.fc_none = nn.Linear(num_features, 1)
        self.fc_inverse = nn.Linear(num_features, 1)
        self.fc_sqrt_inv = nn.Linear(num_features, 1)
        
        # 定义回归头选择网络
        self.selector = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 输出三个回归头的权重
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 使用预训练模型提取特征
        _, _, features = self.feature_extractor(x)
        
        # 对每个回归头计算输出
        ef_none = self.fc_none(features)
        ef_inverse = self.fc_inverse(features)
        ef_sqrt_inv = self.fc_sqrt_inv(features)
        
        # 在选择网络训练阶段计算选择权重
        weights = self.selector(features)
        
        # 计算加权后的最终EF值
        combined_ef = torch.cat((ef_none, ef_inverse, ef_sqrt_inv), dim=1) * weights
        final_ef = combined_ef.sum(dim=1, keepdim=True)
        
        # 返回每个回归头的输出、最终EF值和选择权重
        return ef_none, ef_inverse, ef_sqrt_inv, final_ef, weights
    

def train_model(model, train_dataloader, test_fnames, device, run_dir,mean, std, frames, period, num_epochs=10):
    model.train()
    optimizer_regression_heads = torch.optim.SGD([
        {'params': model.fc_none.parameters()},
        {'params': model.fc_inverse.parameters()},
        {'params': model.fc_sqrt_inv.parameters()},
    ], lr=1e-4)
    
    optimizer_selector = torch.optim.SGD(model.selector.parameters(), lr=1e-4)

    # 定义学习率调度器
    scheduler_regression_heads = StepLR(optimizer_regression_heads, step_size=5, gamma=0.1)
    scheduler_selector = StepLR(optimizer_selector, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for inputs, targets, weights, _, _ in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs, targets = inputs.to(device), targets.to(device)
                weights_none, weights_inverse, weights_sqrt_inv = weights
                weights_none, weights_inverse, weights_sqrt_inv = weights_none.to(device), weights_inverse.to(device), weights_sqrt_inv.to(device)

                targets = targets.unsqueeze(1)
                weights_none = weights_none.unsqueeze(1)
                weights_inverse = weights_inverse.unsqueeze(1)
                weights_sqrt_inv = weights_sqrt_inv.unsqueeze(1)
                
                optimizer_regression_heads.zero_grad()
                optimizer_selector.zero_grad()

                ef_none, ef_inverse, ef_sqrt_inv, final_ef, _ = model(inputs)

                loss_none = F.mse_loss(ef_none, targets) #weighted_mse_loss(ef_none, targets, weights_none)
                loss_inverse = weighted_mse_loss(ef_inverse, targets, weights_inverse)
                loss_sqrt_inv = weighted_mse_loss(ef_sqrt_inv, targets, weights_sqrt_inv)
                total_loss_regression = loss_none + loss_inverse + loss_sqrt_inv
                
                total_loss_regression.backward(retain_graph=True)
                optimizer_regression_heads.step()

                loss_selector = F.mse_loss(final_ef, targets)
                loss_selector.backward()
                optimizer_selector.step()

                tepoch.set_postfix(loss_regression_heads=total_loss_regression.item(), loss_selector=loss_selector.item())

        # 每个epoch结束后更新学习率
        scheduler_regression_heads.step()
        scheduler_selector.step()
        
        # 每个epoch结束后进行测试
        test_epoch_all_moe(model, test_fnames, device, mean, std, frames, period, run_dir, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and Test an Echo Model')
    parser.add_argument('--run_dir', type=str, default="/home/****/EchoNet/EchoMEN/stage2", help='Directory to save outputs')
    parser.add_argument('--weights_path', type=str, default="/stage1/encoder.pt", help='Path to the pretrained weights')
    
    args = parser.parse_args()

    # 加载已经训练好的模型
    model = r2plus1d_18_stage1()
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = args.weights_path  # 使用 argparse 解析的路径
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.fc.bias.data[0] = 55.6
    if device.type == "cuda":
        model = model.to(device)
    
    # 准备好集成模型
    integrated_model = IntegratedModelWithSelectiveTraining(model)
    integrated_model.fc_none.bias.data[0] = 55.6
    integrated_model.fc_inverse.bias.data[0] = 55.6
    integrated_model.fc_sqrt_inv.bias.data[0] = 55.6
    integrated_model.fc_none.weight.data.normal_(0, 0.01)
    integrated_model.fc_inverse.weight.data.normal_(0, 0.01)
    integrated_model.fc_sqrt_inv.weight.data.normal_(0, 0.01)

    # 将模型移动到合适的设备上
    integrated_model = integrated_model.to(device)
    
    
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

    train_dataset = Echo_One_Reweight(split="train", **kwargs)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = Echo_One(split="test", **kwargs)
    test_fnames = test_dataset.fnames 

    # 使用 argparse 解析的 run_dir 路径
    run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)

    # 调用 train_model，确保传递 run_dir 和其他必要参数
    train_model(integrated_model, train_dataloader, test_fnames, device, run_dir, mean, std, frames, period, num_epochs=20)
