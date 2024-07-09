"""Functions for training and running EF prediction."""

import math
import os
import time
import datetime
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import tqdm
import argparse

from utils import loadvideo, bootstrap, latexify, get_mean_and_std
from model import r2plus1d_18_stage1, LDW_SupConLoss
from datasets import Echo

cntrst_loss = LDW_SupConLoss(temperature = 1, base_temperature = 1).to("cuda")
DATA_DIR = "/home/****/EchoNet/a4c-video-dir/"

def run(
    data_dir=None,
    output=None,
    task="EF",

    model_name="r2plus1d_18",
    pretrained=True,
    weights=None,

    run_test=False,
    num_epochs=45,
    lr=1e-4,
    weight_decay=1e-4,
    lr_step_period=15,
    frames=32,
    period=2,
    num_train_patients=None,
    num_workers=4,
    batch_size=20,
    device=None,
    seed=0,

    stgno = 0,
    ctr_w = 0.75
):
    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(model_name, frames, period, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # if os.path.isdir(os.path.join(output, "echonet_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))):
    #     shutil.rmtree(os.path.join(output, "echonet_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))))
    # shutil.copytree("echonet", os.path.join(output, "echonet_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))))

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = r2plus1d_18_stage1(pretrained=pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6
    if device.type == "cuda":
        model = model.to(device)
    # model.to(device)


    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)
    
    mean_dataset = Echo(root=data_dir, split="train")
    print("len of mean dataset", len(mean_dataset))
    mean, std = get_mean_and_std(mean_dataset)
    print("mean std", mean, std)
    kwargs = {"target_type": task,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }


    dataset = {}
    dataset["train"] = Echo(root=data_dir, split="train", **kwargs, pad=12)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = Echo(root=data_dir, split="val", **kwargs)

    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"), worker_init_fn=worker_init_fn)

                loss, loss_reg, loss_ctr, yhat, y, _, _ = run_epoch(model, dataloader, phase == "train", optim, device, stgno = stgno, ctr_w = ctr_w)

                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                              phase,
                                                              loss,
                                                              sklearn.metrics.r2_score(y, yhat),
                                                              time.time() - start_time,
                                                              y.size,
                                                              sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                              sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                              batch_size,
                                                              loss_reg,
                                                              loss_ctr))
                f.flush()
            scheduler.step()

            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': loss,
                'r2': sklearn.metrics.r2_score(y, yhat),
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss_reg < bestLoss:
                torch.save(save, os.path.join(output, "encoder.pt"))
                bestLoss = loss_reg

        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "encoder.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}, R2 {}\n".format(checkpoint["loss"], checkpoint["epoch"], checkpoint["r2"]))
            f.flush()

        if run_test:
            for split in ["test", "val"]:
                np.random.seed(seed)
                torch.manual_seed(seed)

                ds = Echo(root=data_dir, split=split, **kwargs, clips="all")
                yhat, y = test_epoch_all(model, ds, False, None, device, save_all=True, block_size=batch_size, run_dir = output, test_val = split, **kwargs)
                
                f.write("Seed is {} \n".format(seed))
                f.write("{} - {} (all clips, mod) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, *bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("{} - {} (all clips, mod) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, *bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("{} - {} (all clips, mod) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, *tuple(map(math.sqrt, bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                f.flush()

                latexify()
                
                fig = plt.figure(figsize=(3, 3))
                lower = min(y.min(), yhat.min())
                upper = max(y.max(), yhat.max())
                plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
                plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
                plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
                plt.gca().set_aspect("equal", "box")
                plt.xlabel("Actual EF (%)")
                plt.ylabel("Predicted EF (%)")
                plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split)))
                plt.close(fig)

                # Plot AUROC
                fig = plt.figure(figsize=(3, 3))
                plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
                for thresh in [35, 40, 45, 50]:
                    fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat)
                    print(thresh, sklearn.metrics.roc_auc_score(y > thresh, yhat))
                    plt.plot(fpr, tpr)

                plt.axis([-0.01, 1.01, -0.01, 1.01])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_roc.pdf".format(split)))
                plt.close(fig)




def test_epoch_all(model, dataset, train, optim, device, save_all=False, block_size=None, run_dir = None, test_val = None, mean = None, std = None, length = None, period = None, target_type = None):
    model.train(False)

    total = 0 
    total_reg = 0 
    total_ctr = 0

    n = 0      
    s1 = 0     
    s2 = 0     

    yhat = []
    y = []

    if (mean is None) or (std is None) or (length is None) or (period is None):
        assert 1==2, "missing key params"

    max_length = 250

    if run_dir:
        outftcltdir = os.path.join(run_dir, "feat_collect_{}".format(test_val))
        if not os.path.isdir(outftcltdir):
            os.makedirs(outftcltdir)
    
        temp_savefile = os.path.join(run_dir, "temp_inference_{}.csv".format(test_val))

    with torch.set_grad_enabled(False):
        orig_filelist = dataset.fnames

        if os.path.isfile(temp_savefile):
            exist_data = pd.read_csv(temp_savefile)
            exist_file = list(exist_data['fnames'])
            target_filelist = sorted(list(set(orig_filelist) - set(exist_file)))
        else:
            target_filelist = sorted(list(orig_filelist))
            exist_data = pd.DataFrame(columns = ['fnames', 'yhat'])

        if len(target_filelist) <20:
            # 说明是最后一批
            last_block = True
        elif len(target_filelist)==20:
            assert 0, "已经全部跑完了"
            
        for filelistitr_idx in range(len(target_filelist)):
            filelistitr = target_filelist[filelistitr_idx]

            video_path = os.path.join(DATA_DIR, "Videos", filelistitr)
            ### Get data
            video = loadvideo(video_path).astype(np.float32)

            if isinstance(mean, (float, int)):
                video -= mean
            else:
                video -= mean.reshape(3, 1, 1, 1)

            if isinstance(std, (float, int)):
                video /= std
            else:
                video /= std.reshape(3, 1, 1, 1)

            c, f, h, w = video.shape
            if length is None:
                length = f // period
            else:
                length = length

            if max_length is not None:
                length = min(length, max_length)

            if f < length * period:
                video = np.concatenate((video, np.zeros((c, length * period - f, h, w), video.dtype)), axis=1)
                c, f, h, w = video.shape  

            start = np.arange(f - (length - 1) * period)

            reg1 = []
            n_clips = start.shape[0]
            batch = 1
            for s_itr in range(0, start.shape[0], block_size):
                print("{}, processing file {} out of {},  block {} out of {}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), filelistitr_idx, len(target_filelist), s_itr, start.shape[0]), flush=True)
                
                vid_samp = tuple(video[:, s + period * np.arange(length), :, :] for s in start[s_itr: s_itr + block_size])
                X1 = torch.tensor(np.stack(vid_samp))
                X1 = X1.to(device)

                all_output = model(X1)
                reg1.append(all_output[0].detach().cpu().numpy())
            reg1 = np.vstack(reg1)
            reg1_mean = reg1.reshape(batch, n_clips, -1).mean(1)

            exist_data = exist_data.append({'fnames':filelistitr, 'yhat':reg1_mean[0,0]}, ignore_index=True)

            if filelistitr_idx % 20 == 0:
                exist_data.to_csv(temp_savefile, index = False)
                
            if last_block:
                exist_data.to_csv(temp_savefile, index = False)

    
    label_data_path = os.path.join(DATA_DIR, "FileList.csv")
    label_data = pd.read_csv(label_data_path)
    label_data_select = label_data[['FileName','EF']]
    label_data_select.columns = ['fnames','EF']
    with_predict = exist_data.merge(label_data_select, on='fnames')

    predict_out_path = os.path.join(run_dir, "{}_predictions.csv".format(test_val))
    with_predict.to_csv(predict_out_path, index=False)


    return with_predict['yhat'].to_numpy(), with_predict['EF'].to_numpy()


def run_epoch(model, dataloader, train, optim, device, save_all=False, block_size=None, run_dir = None, test_val = None, stgno = 0, ctr_w = 0.75):

    model.train(train)

    total = 0  
    total_reg = 0 
    total_ctr = 0

    n = 0     
    s1 = 0    
    s2 = 0    

    yhat = []
    y = []
    start_frame_record = []
    vidpath_record = []

    if run_dir:
        outftcltdir = os.path.join(run_dir, "feat_collect_{}".format(test_val))
        if not os.path.isdir(outftcltdir):
            os.makedirs(outftcltdir)

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            enum_idx = 0
            for (X1, X2, outcome, outcome_cls, start_frame, video_path) in dataloader:   
                if run_dir:
                    outpath_ftvl = os.path.join(outftcltdir, "featvl_{}".format(enum_idx))
                    outpath_ftv2 = os.path.join(outftcltdir, "featv2_{}".format(enum_idx))
                    outpath_lb = os.path.join(outftcltdir, "lb_{}".format(enum_idx))
                enum_idx = enum_idx + 1

                bsz = outcome.shape[0]
                
                if run_dir:
                    np.save(outpath_lb, outcome.detach().cpu().numpy())

                y.append(outcome.detach().cpu().numpy())
                X1 = X1.to(device)
                X2 = X2.to(device)
                X = torch.cat((X1, X2), dim=0).to('cuda')

                outcome = outcome.to(device)
                outcome_cls = outcome_cls.to(device)

                n_clips = 0
                s1 += outcome.sum()
                s2 += (outcome ** 2).sum()
                    
                all_ouput = model(X)
                outputs = all_ouput[0]
                ctr_feat = all_ouput[1]

                f1, f2 = torch.split(ctr_feat, [bsz, bsz], dim=0)

                if run_dir:
                    np.save(outpath_ftvl, f1.detach().cpu().numpy())
                    np.save(outpath_ftv2, f2.detach().cpu().numpy())

                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                reg1, reg2 = torch.split(outputs, [bsz, bsz], dim=0)                 
              
                yhat.append(reg1.view(-1).to("cpu").detach().numpy())

                if not train:
                    start_frame_record.append(start_frame.view(-1).to("cpu").detach().numpy())
                    vidpath_record.append(video_path)


                loss_reg = torch.nn.functional.mse_loss(reg1.view(-1), outcome)                    
                loss_ctr = cntrst_loss(features, outcome_cls, distance = outcome_cls, normalize_factor=2/7465, scale_factor = 150)

                loss = loss_reg + 0.75 * loss_ctr 
                        
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * outcome.size(0)
                total_reg += loss_reg.item() * outcome.size(0)
                total_ctr += loss_ctr.item() * outcome.size(0)

                n += outcome.size(0)
                pbar.set_postfix_str("{:.2f} {:.2f} {:.2f} ({:.2f}) / {:.2f} {}".format(total / n, total_reg / n, total_ctr / n, loss.item(), s2 / n - (s1 / n) ** 2, n_clips))
                pbar.update()

    if not save_all:
        yhat = np.concatenate(yhat)
        if not train:
            start_frame_record = np.concatenate(start_frame_record)

    y = np.concatenate(y)

    return total / n, total_reg / n, total_ctr / n, yhat, y, start_frame_record, vidpath_record



def main():
    parser = argparse.ArgumentParser(description='Run the training process with specified parameters.')

    parser.add_argument('--data_dir', type=str, default='/home/****/EchoNet/a4c-video-dir/', help='Directory path for data.')
    parser.add_argument('--output', type=str, default='/home/****/EchoNet/EchoMEN/stage1', help='Output directory path.')
    parser.add_argument('--task', type=str, default='EF', help='Task type.')
    parser.add_argument('--model_name', type=str, default='r2plus1d_18', help='Model name.')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model. Default is True.')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights file.')
    parser.add_argument('--run_test', action='store_true', help='Run test after training. Default is False.')
    parser.add_argument('--stgno', type=int, default=0, help='STGNO parameter.')
    parser.add_argument('--num_epochs', type=int, default=45, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay.')
    parser.add_argument('--lr_step_period', type=int, default=15, help='Learning rate step period.')
    parser.add_argument('--frames', type=int, default=32, help='Number of frames.')
    parser.add_argument('--period', type=int, default=2, help='Period.')
    parser.add_argument('--num_train_patients', type=int, default=None, help='Number of training patients.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size.')
    parser.add_argument('--device', type=str, default=None, help='Device to use.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    args = parser.parse_args()

    run(**vars(args))

if __name__ == '__main__':
    main()