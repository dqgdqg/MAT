import numpy as np
import torch
import torch.nn.functional as F

def info_nce_loss(y1, y2, batch_size, temperature):
    y = torch.cat([y1, y2], dim=0)
    y = F.normalize(y, dim=1)

    labels = torch.arange(batch_size).cuda()
    labels = torch.cat([labels, labels], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    sim_mat = torch.matmul(y, y.T)

    mask = torch.eye(labels.shape[0]).bool().cuda()
    labels = labels[~mask].view(labels.shape[0], -1) # 2bs x 2bs-1

    sim_mat = sim_mat[~mask].view(sim_mat.shape[0], -1) # 2bs x 2bs-1

    pos_sim = sim_mat[labels.bool()].view(labels.shape[0], -1)
    neg_sim = sim_mat[~labels.bool()].view(labels.shape[0], -1)

    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
    labels = torch.zeros(logits.shape[0]).long().cuda()

    return logits, labels

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def seconds_to_axis(secs, delta):
    if type(secs) == np.ndarray:
        return (secs / delta).astype(int)
    else:
        return int(secs / delta)

def axis_to_seconds(axis, delta):
    if type(axis) == np.ndarray:
        return (axis * delta).astype(int)
    else:
        return int(axis * delta)

def collate_fn(batch):
    batch = list(zip(*batch)) # x, y, dim_start, dim_end, index_start, index_end, meta
    batch[:-1] = [torch.tensor(np.array(x)) for x in batch[:-1]]
    # batch[:-1] = list(map(lambda x: torch.tensor(np.array(x)), batch[:-1]))
    return batch

def tolerance(pred, t=5):
    i = 0
    delta = 3600 / pred.size
    while i < pred.size - seconds_to_axis(t, delta):
        if pred[i] == 1:
            pred[i+1:i+seconds_to_axis(t, delta)] = 0.
            i = i + seconds_to_axis(t, delta)
        else:
            i = i + 1

    return pred

def load_npz(f):
    meta = dict(np.load(f))
    data = meta["data"]
    
    data = data.astype(np.float32)
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

def load_npz_avg(f, is_abs):
    meta = dict(np.load(f))
    data = meta["data"]
    data = data.astype(np.float32)

    data = moving_average_fast(data.T, 10).T # 1250 x 900
    if is_abs == True:
        data = np.abs(data)

    data -= np.median(data, axis=1, keepdims=True)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    data = data.T

    meta["data"] = data
    meta["file"] = f
    return meta 