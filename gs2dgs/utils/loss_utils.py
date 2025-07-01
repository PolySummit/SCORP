#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
from typing import Literal
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from .image_utils import depth_normalize_

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def isotropic_loss(scaling: torch.Tensor) -> torch.Tensor:
    """
    Computes loss enforcing isotropic scaling for the 3D Gaussians
    Args:
        scaling: scaling tensor of 3D Gaussians of shape (n, 3)
    Returns:
        The computed isotropic loss
    """
    mean_scaling = scaling.mean(dim=1, keepdim=True)
    isotropic_diff = torch.abs(scaling - mean_scaling * torch.ones_like(scaling))
    return isotropic_diff.mean()

def __pearson_depth_loss(depth_src: torch.Tensor, depth_target: torch.Tensor):
    """
    Pearson depth loss function.
    Args:
        depth_src: Source depth tensor. [B, h, w]
        depth_target: Target depth tensor. [B, h, w]
    Returns:
        The computed Pearson depth loss
    """

    src = depth_src - depth_src.mean(dim=(1, 2), keepdim=True)
    target = depth_target - depth_target.mean(dim=(1, 2), keepdim=True)

    src = src / (src.std(dim=(1, 2), keepdim=True) + 1e-6)
    target = target / (target.std(dim=(1, 2), keepdim=True) + 1e-6)

    co = (src * target).mean(dim=(1, 2))
    assert not torch.any(torch.isnan(co))
    return 1 - co.mean()

def random_patch_loss(
    depth_src: torch.Tensor,
    depth_tgt: torch.Tensor,
    # mask: torch.Tensor,
    box_p,
    p_corr,
    loss_func: Literal["l1", "l2", "pearson"] = "l1",
):
    # Randomly select patch, top left corner of the patch (x_0,y_0) has to be 0 <= x_0 <= max_h, 0 <= y_0 <= max_w

    depth_src = depth_src.squeeze()
    depth_tgt = depth_tgt.squeeze()
    # mask = mask.squeeze()

    num_box_h = math.floor(depth_src.shape[0] / box_p)
    num_box_w = math.floor(depth_src.shape[1] / box_p)
    max_h = depth_src.shape[0] - box_p
    max_w = depth_src.shape[1] - box_p

    n_corr = int(p_corr * num_box_h * num_box_w)
    x_0 = torch.randint(0, max_h, size=(n_corr,), device="cuda")
    y_0 = torch.randint(0, max_w, size=(n_corr,), device="cuda")

    _loss = torch.tensor(0.0, device="cuda")

    depth_patches_src = torch.stack([
        depth_src[x_i : x_i + box_p, y_i : y_i + box_p]
        for x_i, y_i in zip(x_0, y_0)
    ], dim=0) # [n_corr, box_p, box_p]
    depth_patches_tgt = torch.stack([
        depth_tgt[x_i : x_i + box_p, y_i : y_i + box_p]
        for x_i, y_i in zip(x_0, y_0)
    ], dim=0)

    if loss_func == "l1":
        _loss = l1_loss(depth_normalize_(depth_patches_src), depth_normalize_(depth_patches_tgt))
    elif loss_func == "l2":
        _loss = l2_loss(depth_normalize_(depth_patches_src), depth_normalize_(depth_patches_tgt))
    elif loss_func == "pearson":
        _loss = __pearson_depth_loss(depth_patches_src, depth_patches_tgt)
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")

    return _loss
