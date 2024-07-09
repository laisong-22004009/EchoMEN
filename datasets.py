import os
import collections
import pandas
import datetime

import numpy as np
import skimage.draw
import torchvision
from utils import loadvideo


class Echo(torchvision.datasets.VisionDataset):
    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None):
        if root is None:
            root = "/home/****/EchoNet/a4c-video-dir/"

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            print(os.path.join(self.root, "FileList.csv"))

            with open(os.path.join(self.root, "FileList.csv")) as f:
                data = pandas.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            # 打印data的信息
            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            
            data["EF_bkt"] = data["EF"]//0.05  # // 0.02 // 0.01
            EF_freq = data['EF_bkt'].value_counts(dropna=False).rename_axis('EF_bkt_key').reset_index(name='counts')
            EF_freq = EF_freq.sort_values(by=['EF_bkt_key']).reset_index()

            EF_dict = {}

            for key_itr_idx in range(len(EF_freq['EF_bkt_key'])):
                if key_itr_idx == 0:
                    EF_dict[EF_freq['EF_bkt_key'][key_itr_idx]] = EF_freq['counts'][key_itr_idx]
                else:
                    EF_dict[EF_freq['EF_bkt_key'][key_itr_idx]] = EF_dict[EF_freq['EF_bkt_key'][key_itr_idx-1]] + EF_freq['counts'][key_itr_idx]
            
            for key_itr_idx in range(len(EF_freq['EF_bkt_key'])):
                EF_dict[EF_freq['EF_bkt_key'][key_itr_idx]] = EF_dict[EF_freq['EF_bkt_key'][key_itr_idx]] - EF_freq['counts'][key_itr_idx]/2

            data['EF_CLS'] = data["EF_bkt"].apply(lambda x: EF_dict[x])

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            self.fnames = [fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix
            
            self.outcome = data.values.tolist()


            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]
            

    def __getitem__(self, index):
        if self.split == "EXTERNAL_TEST":
            video_path = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video_path = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video_path = os.path.join(self.root, "Videos", self.fnames[index])

        video = loadvideo(video_path).astype(np.float32)

        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        c, f, h, w = video.shape
        if self.length is None:
            length = f // self.period
        else:
            length = self.length

        if self.max_length is not None:
            length = min(length, self.max_length)

        if f < length * self.period:
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  

        if self.clips == "all":
            start = np.arange(f - (length - 1) * self.period)
        else:
            start = np.random.choice(f - (length - 1) * self.period, self.clips)


        target = []
        target_cls = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                target.append(np.int(self.frames[key][-1]))
            elif t == "SmallIndex":
                target.append(np.int(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target.append(mask)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))
                    target_cls.append(np.float32(self.outcome[index][self.header.index('EF_CLS')]))

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)
        if target_cls !=[]:
            target_cls = tuple(target_cls) if len(target_cls) > 1 else target_cls[0]

        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        if self.pad is not None:

            jit1 = np.random.random()*0.1
            jit2 = np.random.random()*0.1

            # video1 = video + jit1
            # video2 = video + jit2

            video1 = video.copy()
            video2 = video.copy()


            c, l, h, w = video.shape

            temp1 = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp1[:, :, self.pad:-self.pad, self.pad:-self.pad] = video1

            temp2 = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp2[:, :, self.pad:-self.pad, self.pad:-self.pad] = video2

            i1, j1 = np.random.randint(0, 2 * self.pad, 2)
            i2, j2 = np.random.randint(0, 2 * self.pad, 2)

            video1 = temp1[:, :, i1:(i1 + h), j1:(j1 + w)]
            video2 = temp2[:, :, i2:(i2 + h), j2:(j2 + w)]


        else:
            video1 = video.copy()
            video2 = video.copy()

        return video1, video2, target, target_cls, start, video_path

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
   
   
   
 
class Echo_One(torchvision.datasets.VisionDataset):
    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None):
        if root is None:
            root = "/home/****/EchoNet/a4c-video-dir/"

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            print(os.path.join(self.root, "FileList.csv"))

            with open(os.path.join(self.root, "FileList.csv")) as f:
                data = pandas.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            # 打印data的信息
            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            
            data["EF_bkt"] = data["EF"]//0.05  # // 0.02 // 0.01
            EF_freq = data['EF_bkt'].value_counts(dropna=False).rename_axis('EF_bkt_key').reset_index(name='counts')
            EF_freq = EF_freq.sort_values(by=['EF_bkt_key']).reset_index()

            EF_dict = {}

            for key_itr_idx in range(len(EF_freq['EF_bkt_key'])):
                if key_itr_idx == 0:
                    EF_dict[EF_freq['EF_bkt_key'][key_itr_idx]] = EF_freq['counts'][key_itr_idx]
                else:
                    EF_dict[EF_freq['EF_bkt_key'][key_itr_idx]] = EF_dict[EF_freq['EF_bkt_key'][key_itr_idx-1]] + EF_freq['counts'][key_itr_idx]
            
            for key_itr_idx in range(len(EF_freq['EF_bkt_key'])):
                EF_dict[EF_freq['EF_bkt_key'][key_itr_idx]] = EF_dict[EF_freq['EF_bkt_key'][key_itr_idx]] - EF_freq['counts'][key_itr_idx]/2

            data['EF_CLS'] = data["EF_bkt"].apply(lambda x: EF_dict[x])

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            self.fnames = [fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix
            
            self.outcome = data.values.tolist()


            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]
            

    def __getitem__(self, index):
        if self.split == "EXTERNAL_TEST":
            video_path = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video_path = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video_path = os.path.join(self.root, "Videos", self.fnames[index])

        video = loadvideo(video_path).astype(np.float32)

        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        c, f, h, w = video.shape
        if self.length is None:
            length = f // self.period
        else:
            length = self.length

        if self.max_length is not None:
            length = min(length, self.max_length)

        if f < length * self.period:
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  

        if self.clips == "all":
            start = np.arange(f - (length - 1) * self.period)
        else:
            start = np.random.choice(f - (length - 1) * self.period, self.clips)


        target = []
        target_cls = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                target.append(np.int(self.frames[key][-1]))
            elif t == "SmallIndex":
                target.append(np.int(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target.append(mask)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))
                    target_cls.append(np.float32(self.outcome[index][self.header.index('EF_CLS')]))

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)
        if target_cls !=[]:
            target_cls = tuple(target_cls) if len(target_cls) > 1 else target_cls[0]

        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        
        if self.clips == 1:
            video = video[0]  #只选随机的第一个clip 但clip本身也是随机的 所以还是等于随机选了一个clip
        else:
            video = np.stack(video)

        if self.pad is not None:


            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video

            i1, j1 = np.random.randint(0, 2 * self.pad, 2)

            video = temp[:, :, i1:(i1 + h), j1:(j1 + w)]


        return video, target, start, video_path

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

class Echo_One_Reweight(torchvision.datasets.VisionDataset):
    def __init__(self, root=None, split="train", target_type="EF", mean=0., std=1.,
                 length=16, period=2, max_length=250, clips=1, pad=None, noise=None):
        if root is None:
            root = "/home/****/EchoNet/a4c-video-dir/"
        super().__init__(root)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise

        # Load dataset information
        with open(os.path.join(self.root, "FileList.csv")) as f:
            data = pandas.read_csv(f)
        data["Split"] = data["Split"].str.upper()
        if self.split != "ALL":
            data = data[data["Split"] == self.split]

        self.fnames = [fn + ".avi" for fn in data["FileName"].tolist() if os.path.splitext(fn)[1] == ""]
        self.ef = data["EF"].tolist()

        # Prepare weights
        self.weights_none, self.weights_inverse, self.weights_sqrt_inv = self._prepare_weights()
    
    def _prepare_weights(self):
        # Calculate bin counts
        bins = np.linspace(0, 100, 101)  # 100 bins for EF values from 0 to 100
        bin_counts, _ = np.histogram(self.ef, bins=bins)
        bin_counts = np.clip(bin_counts, 80, 1000)  # Avoid division by zero

        # Calculate weights for each strategy
        weights_none = np.ones_like(self.ef)
        weights_inverse = 1 / np.interp(self.ef, (bins[:-1] + bins[1:]) / 2, bin_counts)
        weights_sqrt_inv = 1 / np.sqrt(np.interp(self.ef, (bins[:-1] + bins[1:]) / 2, bin_counts))

        # Normalize weights
        weights_none /= np.mean(weights_none)
        weights_inverse /= np.mean(weights_inverse)
        weights_sqrt_inv /= np.mean(weights_sqrt_inv)

        return weights_none, weights_inverse, weights_sqrt_inv



    def __getitem__(self, index):
        video_path = os.path.join(self.root, "Videos", self.fnames[index])
        video = loadvideo(video_path).astype(np.float32)
        
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        c, f, h, w = video.shape
        if self.length is None:
            length = f // self.period
        else:
            length = self.length

        if self.max_length is not None:
            length = min(length, self.max_length)

        if f < length * self.period:
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  

        if self.clips == "all":
            start = np.arange(f - (length - 1) * self.period)
        else:
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        
        if self.clips == 1:
            video = video[0]  #只选随机的第一个clip 但clip本身也是随机的 所以还是等于随机选了一个clip
        else:
            video = np.stack(video)

        if self.pad is not None:
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video

            i1, j1 = np.random.randint(0, 2 * self.pad, 2)

            video = temp[:, :, i1:(i1 + h), j1:(j1 + w)]

        target = np.float32(self.ef[index])
        weights = (np.float32(self.weights_none[index]),
                   np.float32(self.weights_inverse[index]),
                   np.float32(self.weights_sqrt_inv[index]))
        
        return video, target, weights, start, video_path
    
    def __len__(self):
        return len(self.fnames)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)

if __name__ == "__main__":
    dataset = Echo_One(split="train")
    for i in range(len(dataset)):
        video, target, start, video_path = dataset[i]
        print(video.shape, target, start, video_path)
        
    