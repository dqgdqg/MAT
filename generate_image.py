# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
import cv2
import pyspng
import glob
import os
import re
import random
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import legacy
from datasets.mask_generator_512 import RandomMask
from networks.mat import Generator
import matplotlib.pyplot as plt

import ot
import scipy.stats
from sklearn.metrics import roc_auc_score

from IPython import embed

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        if name not in src_tensors:
            embed()

        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dpath', help='the path of the input image', required=True)
@click.option('--mpath', help='the path of the mask')
@click.option('--resolution', type=int, help='resolution of input image', default=512, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    dpath: str,
    mpath: Optional[str],
    resolution: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
):
    """
    Generate images using pretrained network pickle.
    """
    seed = 240  # pick up a random number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(f'Loading data from: {dpath}')
    img_list = sorted(glob.glob(dpath + '/*.png') + glob.glob(dpath + '/*.jpg'))

    if mpath is not None:
        print(f'Loading mask from: {mpath}')
        mask_list = sorted(glob.glob(mpath + '/*.png') + glob.glob(mpath + '/*.jpg'))
        assert len(img_list) == len(mask_list), 'illegal mapping'

    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
    net_res = 512 if resolution > 512 else resolution
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=True)

    os.makedirs(outdir, exist_ok=True)

    # no Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    def read_image(image_path):
        with open(image_path, 'rb') as f:
            if pyspng is not None and image_path.endswith('.png'):
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
            image = np.repeat(image, 3, axis=2)
        image = image.transpose(2, 0, 1) # HWC => CHW
        image = image[:3]
        return image

    def to_image(image, lo, hi):
        image = np.asarray(image, dtype=np.float32)
        image = (image - lo) * (255 / (hi - lo))
        image = np.rint(image).clip(0, 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image

    if resolution != 512:   
        noise_mode = 'random'
    with torch.no_grad():
        for i, ipath in enumerate(img_list):
            iname = os.path.basename(ipath).replace('.jpg', '.png')
            print(f'Prcessing: {iname}')
            image = read_image(ipath)
            image = (torch.from_numpy(image).float().to(device) / 127.5 - 1).unsqueeze(0)

            if mpath is not None:
                mask = cv2.imread(mask_list[i], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0).unsqueeze(0)
            else:
                mask = RandomMask(resolution) # adjust the masking ratio by using 'hole_range'
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)

            z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
            output = G(image, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            output = output[0].cpu().numpy()
            PIL.Image.fromarray(output, 'RGB').save(f'{outdir}/{iname}')

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dpath', help='the path of the input tensors', required=True)
@click.option('--ppath', help='the path of the picks', required=True)
@click.option('--resolution', type=int, help='resolution of input image', default=512, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_tensors(
    ctx: click.Context,
    network_pkl: str,
    dpath: str, # Data path
    ppath: str,
    # mpath: Optional[str], # Mask Path - No need
    resolution: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
):
    """
    Generate images using pretrained network pickle.
    """
    seed = 240  # pick up a random number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
        'places256': dict(ref_gpus=8,  kimg=50000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8),
        'places512': dict(ref_gpus=8,  kimg=50000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8),
        'celeba512': dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8),
    }

    cfg = 'auto'
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        res = 128
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32

    G_kwargs = dnnlib.EasyDict(class_name='networks.mat.Generator', \
        z_dim=512, w_dim=512, c_dim=0, img_resolution=128, img_channels=1, \
        mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    G_kwargs.synthesis_kwargs.channel_base = int(spec.fmaps * 32768)
    G_kwargs.synthesis_kwargs.channel_max = 512
    G_kwargs.mapping_kwargs.num_layers = spec.map


    print(f'Loading data from: {dpath}')
    # img_list = sorted(glob.glob(dpath + '/*.png') + glob.glob(dpath + '/*.jpg'))
    tensors = torch.load(dpath)
    tensors = tensors.unsqueeze(1) # N, 1, 128, 128

    print(f'Loading picks from: {ppath}')
    # img_list = sorted(glob.glob(dpath + '/*.png') + glob.glob(dpath + '/*.jpg'))
    picks = torch.load(ppath)
    picks = picks.unsqueeze(1) # N, 1, 128, 128

    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
    net_res = 512 if resolution > 512 else resolution

    # G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=1, ).to(device).eval().requires_grad_(False)
    G = dnnlib.util.construct_class_by_name(**G_kwargs).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=True)

    os.makedirs(outdir, exist_ok=True)

    # no Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    def numpy_to_image(array, filename):
        fig = plt.figure()
        plt.imshow(array, aspect='auto', vmin=-2.0, vmax=2.0, cmap="seismic")
        plt.savefig(filename)
        plt.cla()
        plt.clf()
        plt.close()

    if resolution != 512:   
        noise_mode = 'random'

    mask_ratio = 0.2
    mask_middle = torch.ones((resolution, resolution))
    left = int(resolution * (0.5 - mask_ratio * 0.5))
    right = int(resolution * (0.5 + mask_ratio * 0.5))
    mask_middle[:, left:right] = 0.

    mask_right = torch.ones((resolution, resolution))
    left = int(resolution * mask_ratio)
    mask_right[:, -left:] = 0.

    mask_middle = mask_middle.unsqueeze(0).unsqueeze(0).to(device)
    mask_right = mask_right.unsqueeze(0).unsqueeze(0).to(device)

    # 

    diff_list = []
    diff_abs_list = []
    diff_emd_list = []
    sum_list = []

    has_pick_list = []
    
    with torch.no_grad():
        for i, tensor in enumerate(tensors):
            pick = picks[i]
            print(f'Prcessing: {i}')
            tensor = tensor.unsqueeze(0).to(device)
            has_pick = (pick.sum() > 0).item()

            #### Middle 
            z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
            output_middle = G(tensor, mask_middle, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            output_middle = output_middle.squeeze().cpu().numpy()

            input_middle = tensor.squeeze().cpu().numpy()

            numpy_to_image(input_middle, os.path.join(outdir, 'input_middle_{}.png'.format(i)))
            numpy_to_image(output_middle, os.path.join(outdir, 'output_middle_{}.png'.format(i)))
            numpy_to_image((output_middle-input_middle), os.path.join(outdir, 'diff_middle_{}.png'.format(i)))
            


            #### Right 
            z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
            output_right = G(tensor, mask_right, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            output_right = output_right.squeeze().cpu().numpy()

            input_right = tensor.squeeze().cpu().numpy()

            numpy_to_image(input_right, os.path.join(outdir, 'input_right_{}.png'.format(i)))
            numpy_to_image(output_right, os.path.join(outdir, 'output_right_{}.png'.format(i)))
            numpy_to_image((output_right-input_right), os.path.join(outdir, 'diff_right_{}.png'.format(i)))

            mask_right_invert = (1 - mask_right).squeeze().cpu().numpy()
            
            diff = (output_right - input_right).sum()
            diff_abs = (np.abs(output_right) - np.abs(input_right)).sum()
            diff_emd = scipy.stats.wasserstein_distance(input_right.flatten(), output_right.flatten())
            summ = (input_right * mask_right_invert).sum()

            diff_list.append(diff)
            diff_abs_list.append(diff_abs)
            diff_emd_list.append(diff_emd)
            sum_list.append(summ)
            has_pick_list.append(has_pick)

    auc_diff = roc_auc_score(has_pick_list, diff_list)
    auc_diff_abs = roc_auc_score(has_pick_list, diff_abs_list)
    auc_emd = roc_auc_score(has_pick_list, diff_emd_list)
    auc_sum = roc_auc_score(has_pick_list, sum_list)

    print('AUC_DIFF: {}\nAUC_DIFF_ABS: {}\nAUC_EMD: {}\nAUC_SUM: {}'.format(auc_diff, auc_diff_abs, auc_emd, auc_sum))
    # roc_auc_score(has_pick_list, diff_list)
    



if __name__ == "__main__":
    generate_tensors() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
