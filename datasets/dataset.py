import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
from datetime import datetime, timedelta
# - timedelta(days=1)
from IPython import embed
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None 

class Earthquake_SSL(Dataset):
    def __init__(self, data_raw, start_time, end_time, window_size):
        points_per_sec = data_raw.shape[1] / 3600
        dim_start = int(start_time * points_per_sec)
        dim_end = int(end_time * points_per_sec)

        self.data = data_raw[:, dim_start: dim_end]
        self.length = self.data.shape[1] - window_size
        self.window_size = window_size

    def amplitude_scale(self, x, scale=2):
        x_p = np.copy(x)
        k = np.random.uniform(low=1.0, high=scale)
        s = np.random.binomial(1, 0.5)
        if s == 0:
            k = 1/k
        
        x_p = x_p * k

        return x_p
    
    def masking(self, x, mask_ratio=0.15):
        x_p = np.copy(x)
        k = np.random.uniform(low=0.1, high=mask_ratio)
        
        mask_size = k * x.shape[1]
        dim_start = int(np.random.uniform(low=0.0, high=(1-k)) * x.shape[1])
        dim_end = int(dim_start + mask_size)

        if x_p[:, dim_start: dim_end].size == 0:
            print(k, mask_size, dim_start, dim_end)

        x_p[:, dim_start: dim_end] = x_p[:, dim_start: dim_end].mean()

        return x_p

    def DC_shift(self, x, shift_ratio=0.15):
        x_p = np.copy(x)
        k = np.random.uniform(low=-shift_ratio, high=shift_ratio)

        x_std = x.std()

        shift_value = x_std * k
        x_p = x_p + shift_value

        return x_p

    def gaussian_noise(self, x, sigma_ratio=0.15):
        x_p = np.copy(x)
        k = np.random.uniform(low=0., high=sigma_ratio)

        x_std = x.std()
        n_std = x_std * k

        n = np.random.normal(0.0, n_std)
        x_p = x_p + n

        return x_p

    def __getitem__(self, start_index):
        end_index = start_index + self.window_size
        x = self.data[:, start_index: end_index]

        t_list = np.random.choice(4, 2)
        x_list = []

        for t in t_list:
            if t == 0:
                x_list.append(self.amplitude_scale(x))
            elif t == 1:
                x_list.append(self.masking(x))
            elif t == 2:
                x_list.append(self.DC_shift(x))
            elif t == 3:
                x_list.append(self.gaussian_noise(x))
        
        return x_list[0], x_list[1]
    
    def __len__(self):
        return self.length
        
class Earthquake_Anomaly(Dataset):
    def __init__(self, data_raw, meta='None', start_time=0, end_time=30, window_size=100, total_time=3600, mask_ratio=0.05, test_mode=False, threshold_mode=False, all=False):
        if all == False:
            points_per_sec = data_raw.shape[1] / total_time
            dim_start = int(start_time * points_per_sec)
            dim_end = int(end_time * points_per_sec)
        else:
            dim_start = 0
            dim_end = data_raw.shape[1]

        self.data = data_raw[:, dim_start: dim_end] # 1250 x time
        self.window_size = window_size
        self.mask_ratio = mask_ratio
        self.dim_start = dim_start
        self.dim_end = dim_end
        self.test_mode = test_mode
        self.meta = meta
        self.threshold_mode = threshold_mode

        if threshold_mode == True:
            self.length = (self.data.shape[1] - window_size) // window_size + 1
        else:
            self.length = self.data.shape[1] - window_size + 1

        self.label = np.zeros(self.data.shape[1], dtype=np.float32)

        if test_mode == True:
            self.p_picks = np.rint(self.meta['p_picks'][:,1]/10).astype(int)
            self.s_picks = np.rint(self.meta['s_picks'][:,1]/10).astype(int)
            self.label[self.p_picks.min():self.p_picks.max()] = 1.
            self.label[self.s_picks.min():self.s_picks.max()] = 1.

    def __getitem__(self, index_start):
        if self.threshold_mode == True:
            index_start = index_start * self.window_size
        index_end = index_start + self.window_size

        mask_size = int(self.window_size * self.mask_ratio)
        mask_start = index_end - mask_size # index_start -> mask_start -> index_end

        x = self.data[:, index_start: mask_start] # D x L
        y = self.data[:, mask_start: index_end]

        return x, y, index_start, index_end, mask_size, self.meta
    
    def __len__(self):
        return self.length

class Qiang_Anomaly(Dataset):
    def __init__(self, setting='train', window_size=100, mask_ratio=0.1, test_mode=False, threshold_mode=False, is_abs=False):
        # points_per_sec = data_raw.shape[1] / total_time
        self.is_abs = is_abs
        dir_path = './data/DAS/{}_npz'.format(setting)
        self.data_raw = self.load_data(dir_path) # nnn x [(1250 x time), meta]

        self.data = list(map(lambda x: Earthquake_Anomaly(
            x[0],
            meta = x[1],
            window_size = window_size,
            mask_ratio = mask_ratio,
            test_mode = test_mode,
            threshold_mode = threshold_mode,
            all = True,
            ), self.data_raw)) 
        self.num_data = len(self.data_raw)
        self.num_sample_per_data = self.data[0].__len__()

        self.num_sample = self.num_data * self.num_sample_per_data
        
    
    def __getitem__(self, index):
        index_data = int(index / self.num_sample_per_data)
        index_sample = index % self.num_sample_per_data

        return self.data[index_data].__getitem__(index_sample)
    
    def __len__(self):
        return self.num_sample

    def load_data(self, dir_path):
        data_raw = []
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for name in files:
                f = os.path.join(dir_path, name)
                meta = self.load_npz(f)
                item = []
                item.append(meta.pop('data'))
                item.append(meta)
                data_raw.append(item)
        
        return data_raw

    def load_npz(self, f):
        try:
            meta = dict(np.load(f))
        except:
            embed()
        data = meta["data"]
        data = data.astype(np.float32)
        
        data = moving_average_fast(data.T, 10).T # 1250 x 900
        if self.is_abs == True:
            data = np.abs(data)

        data -= np.median(data, axis=1, keepdims=True)
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0)
        data = data.T

        meta["data"] = data
        meta["file"] = f

        return meta

def moving_average_fast(x, w):
    x = np.mean(x.reshape(x.shape[0], -1, w), -1)
    return x

def collate_fn(batch):
    batch = list(zip(*batch)) # x, y, dim_start, dim_end, index_start, index_end, meta
    # batch[:-1] = list(map(lambda x: torch.tensor(np.array(x)), batch[:-1]))
    batch[:-1] = [torch.tensor(np.array(x)) for x in batch[:-1]]
    return batch

if __name__ == '__main__':
    # data_raw = np.load('data/data.npy')
    # example = Earthquake_Anomaly(data_raw, 0, 2000, 100)

    data_train = Qiang_Anomaly(setting='train', window_size=200, mask_ratio=0.2, test_mode=False)
    data_valid = Qiang_Anomaly(setting='valid', window_size=200, mask_ratio=0.2, test_mode=False)
    data_test = Qiang_Anomaly(setting='test', window_size=200, mask_ratio=0.2, test_mode=False)

    data_train.__getitem__(0)
    
    train_np = []
    for j in range(len(data_train.data)):
        data_sub = data_train.data[j].data
        train_np.append(data_sub)

    train_np = np.concatenate(train_np, axis=1).flatten()

    valid_np = []
    for j in range(len(data_valid.data)):
        data_sub = data_valid.data[j].data
        valid_np.append(data_sub)

    valid_np = np.concatenate(valid_np, axis=1).flatten()

    test_np = []
    for j in range(len(data_test.data)):
        data_sub = data_test.data[j].data
        test_np.append(data_sub)

    test_np = np.concatenate(test_np, axis=1).flatten()

    plt.hist(train_np, bins=100, alpha=0.45, color='red', density=True)
    plt.hist(valid_np, bins=100, alpha=0.45, color='blue', density=True)
    plt.hist(test_np, bins=100, alpha=0.45, color='green', density=True)

    
    plt.legend(['train','valid','test'])
    
    plt.savefig('hist.pdf')
        
    
    dataloader_train = DataLoader(data_train, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)

