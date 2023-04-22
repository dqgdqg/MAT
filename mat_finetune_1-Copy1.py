#!/usr/bin/env python
# coding: utf-8

# In[4]:


from tracemalloc import start
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from datasets.dataset import *
# from models.Autoformer import *(我h'h'h)
from tqdm import tqdm
import argparse
import random
import math

from six.moves import cPickle as pickle

import scipy.stats
from scipy.signal import convolve2d
from scipy.signal import find_peaks
from sklearn.metrics import *

from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import os

from my_utils import *
from networks.mat import Finetune

from sklearn.metrics import roc_auc_score

from IPython import embed

import matplotlib.pyplot as plt
# %matplotlib inline


# In[5]:


cuda_device = '1'
save_path = '/data/rech/dingqian/model_das/finetune_thin_1e-3'

scratch = True
ft_path = ['/data/rech/dingqian/data_das/finetune.npy', 
          '/data/rech/dingqian/data_das/finetune_test.npy']

if not os.path.exists(save_path):
    os.makedirs(save_path)


# In[6]:


def check_mem(cuda_device):
    devices_info = os.popen('"/opt/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    print(devices_info)
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max(0, max_mem - used)
    print(block_mem)
    x = torch.FloatTensor(256,1024,block_mem).to(f'cuda:{cuda_device}')
    del x
    
occumpy_mem(cuda_device)


# In[3]:


with open(ft_path[0], 'rb') as f:
    x_ft_train, style_ft_train, skip_ft_train = pickle.load(f)
with open(ft_path[1], 'rb') as f:
    x_ft_test, style_ft_test, skip_ft_test = pickle.load(f)


# In[4]:


picks_train = torch.load('/data/rech/dingqian/data_das/picks_train.pt')
patches_train = torch.load('/data/rech/dingqian/data_das/patches_train.pt')

picks_test = torch.load('/data/rech/dingqian/data_das/picks_test.pt')
patches_test = torch.load('/data/rech/dingqian/data_das/patches_test.pt')


# In[5]:


'''finetune_new_train = [x_ft_train, style_ft_train]
finetune_new_test = [x_ft_test, style_ft_test]
with open('/data/rech/dingqian/data_das/finetune_new_train.npy', 'wb') as f:
    pickle.dump(finetune_new_train, f)

with open('/data/rech/dingqian/data_das/finetune_new_test.npy', 'wb') as f:
    pickle.dump(finetune_new_test, f)'''


# In[6]:


# Convert picks to picks_filtered

def picks_filtered(picks, patches):
    picks_filtered = []
    patches_filtered = []

    for i, pick in enumerate(picks):
        if pick.sum().item() > 0:
            picks_filtered.append(pick)
            patches_filtered.append(patches[i])
    
    return torch.stack(picks_filtered), torch.stack(patches_filtered)


# In[7]:


picks_train, patches_train = picks_filtered(picks_train, patches_train)
picks_test, patches_test = picks_filtered(picks_test, patches_test)


# In[1]:


## picks_filtered to gaussian

def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    # return 1 / (2*math.pi*sx*sy) * torch.exp(-((x - mx)**2 / (2*sx**2) + (y - my)**2 / (2*sy**2)))
    return torch.exp(-((x - mx)**2 / (2*sx**2) + (y - my)**2 / (2*sy**2)))

def tensor_to_hot(pick_oh_mat):
    
    pick_pos = torch.stack(torch.where(pick_oh_mat == 1), axis=1)
    
    h, w = 128, 128
    x = torch.linspace(0, h-1, h)
    y = torch.linspace(0, w-1, w)
    x, y = torch.meshgrid(x, y)

    z = torch.zeros(h, w)
    for x0, y0 in pick_pos:
        # z = torch.max(z, gaussian_2d(x, y, mx=x0, my=y0, sx=h/10, sy=w/10))
        z = (z + gaussian_2d(x, y, mx=x0, my=y0, sx=h/100, sy=w/200)).clip(max=1.)
    
    return z
    


# In[9]:


### Get labels

def get_labels(picks_filtered):
    labels = []
    for i, pick in enumerate(tqdm(picks_filtered)):
        pick_oh = F.one_hot(pick.long(), 3)
        pick_p = tensor_to_hot(pick_oh[:, :, 1])
        pick_s = tensor_to_hot(pick_oh[:, :, 2])
        
        pick_n = (1 - pick_p - pick_s).clip(min=0.)
        
        pick_new = torch.stack([pick_n, pick_p, pick_s])
        labels.append(pick_new)
    
    return torch.stack(labels)


# In[10]:


labels_train = get_labels(picks_train)
labels_test = get_labels(picks_test)

    


# In[11]:


labels_train.shape, labels_test.shape


# In[12]:


### Write a train function for each epoch

def train_loop(train_loader, model, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for i, (x, style, skip, labels, picks) in enumerate(tqdm(train_loader)):
        x, style, skip, labels, picks = x.to(device), style.to(device), skip.to(device), labels.to(device), picks.to(device)
        optimizer.zero_grad()
        _, y = model(x, style, skip)
        y = y.softmax(dim=1)
        
        label_indices = torch.where(picks.sum(-1) > 0)
        labels = labels[label_indices[0], :, label_indices[1], :]
        y = y[label_indices[0], :, label_indices[1], :]

        loss = criterion(y, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss / len(train_loader)


# In[13]:


def test_loop(test_loader, model, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (x, style, skip, labels, picks) in enumerate(test_loader):
            x, style, skip, labels, picks = x.to(device), style.to(device), skip.to(device), labels.to(device), picks.to(device)
            _, y = model(x, style, skip)
            y = y.softmax(dim=1)
            
            label_indices = torch.where(picks.sum(-1) > 0)
            labels = labels[label_indices[0], :, label_indices[1], :]
            y = y[label_indices[0], :, label_indices[1], :]
            
            loss = criterion(y, labels)
            test_loss += loss.item()
    
    return test_loss / len(test_loader)


# In[14]:


# Output a sample image with predicted labels and ground truth labels on training set

# get the first batch of data in train_loader

def plot_sample(model, data_loader, epoch, prefix, device):
    model.eval()
    with torch.no_grad():
        x, style, skip, labels, picks = next(iter(data_loader))
        x, style, skip, labels, picks = x.to(device), style.to(device), skip.to(device), labels.to(device), picks.to(device)
        _, y = model(x, style, skip)
        y = y.softmax(dim=1)

        y = y.cpu().numpy()
        labels = labels.cpu().numpy()
        x = x.cpu().numpy()

        fig, ax = plt.subplots(2, 2, figsize=(15, 5))
        ax[0][0].imshow(y[0][1], aspect='auto', vmin=-0, vmax=1.0, cmap="seismic")
        ax[0][1].imshow(labels[0][1], aspect='auto', vmin=-.0, vmax=1.0, cmap="seismic")

        ax[1][0].imshow(y[0][2], aspect='auto', vmin=-0, vmax=1.0, cmap="seismic")
        ax[1][1].imshow(labels[0][2], aspect='auto', vmin=-.0, vmax=1.0, cmap="seismic")

        # save fig to file
        fig.savefig(f'{save_path}/{prefix}_{epoch}.png', dpi=300)

        # clear fig ax
        plt.close(fig)

        print('Max y[1]: ', y[0][1].max(), 'Max y[2]: ', y[0][2].max())
        print('Max labels[1]: ', labels[0][1].max(), 'Max labels[2]: ', labels[0][2].max())

        ### x has 180 samples. using matplotlib to plot x in a grid of size 12x15

        fig, ax = plt.subplots(12, 15, figsize=(15, 12))
        for i in range(12):
            for j in range(15):
                ax[i][j].imshow(x[0][i*15+j], aspect='auto', vmin=-1.0, vmax=1.0, cmap="viridis")
                ax[i][j].axis('off')
                
        # save fig to file
        fig.savefig(f'{save_path}/{prefix}_x_{epoch}.png', dpi=300)

        # clear fig ax
        plt.close(fig)
    


# In[15]:


# Convert numpy data to pytorch dataset

x_ft_train = torch.from_numpy(x_ft_train).float()
style_ft_train = torch.from_numpy(style_ft_train).float()
skip_ft_train = torch.from_numpy(skip_ft_train).float()

x_ft_test = torch.from_numpy(x_ft_test).float()
style_ft_test = torch.from_numpy(style_ft_test).float()
skip_ft_test = torch.from_numpy(skip_ft_test).float()


# In[16]:


# Create dataset for training and testing
# input data: x_ft, style_ft, skip_ft
# output data: labels

train_dataset = TensorDataset(x_ft_train, style_ft_train, skip_ft_train, labels_train, picks_train)
test_dataset = TensorDataset(x_ft_test, style_ft_test, skip_ft_test, labels_test, picks_test)

# Create dataloader for training and testing

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[17]:


x_ft_test.shape, style_ft_test.shape, skip_ft_test.shape, labels_test.shape, picks_test.shape


# In[18]:


labels_test[0][1].sum(1)


# In[19]:


picks_test.shape


# In[20]:


labels_test[0][:, :, torch.where(picks_test[0].sum(-1) > 0)[0]]


# In[21]:


torch.where(picks_test[0].sum(1) > 0)


# In[ ]:


# Define model
import importlib
import networks.mat as mat
importlib.reload(mat)

device = torch.device(f'cuda:{cuda_device}')

model = mat.Finetune(img_channels=3).to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Define loss function as Softmax Cross Entropy
criterion = nn.CrossEntropyLoss()

# Define loss function as MSE loss

# criterion = nn.MSELoss()


# In[ ]:


# Train model and save best model on test set
try:
    model.load_state_dict(torch.load(f'{save_path}/last_model.pth'))
    epoch_start = 6730
except:
    epoch_start = 0

epochs = 4000
best_loss = np.inf
for epoch in range(epoch_start, epoch_start + epochs):
    train_loss = train_loop(train_loader, model, optimizer, criterion, device)
    test_loss = test_loop(test_loader, model, criterion, device)
    print(f'Epoch {epoch}: train loss {train_loss:.4f}, test loss {test_loss:.4f}')

    if epoch % 1 == 0:
        plot_sample(model, train_loader, epoch, 'train', device)
        plot_sample(model, test_loader, epoch, 'test', device)
    
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), f'{save_path}/best_model.pth')
    
    torch.save(model.state_dict(), f'{save_path}/last_model.pth')


# In[ ]:


embed()

