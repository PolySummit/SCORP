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

from typing import NamedTuple
import torch
from torch import nn
import numpy as np
from PIL import Image

from ..utils.general_utils import PILtoTorch
from ..utils.graphics_utils import getWorld2View2, getProjectionMatrix

class GroundTruth(NamedTuple):
    image: torch.Tensor
    alpha: torch.Tensor | None
    depth_cam: torch.Tensor | None
    depth_est: torch.Tensor | None

class Camera(nn.Module):

    preload = False

    def __init__(
        self,
        colmap_id,
        R: np.ndarray,
        T: np.ndarray,
        FoVx,
        FoVy,
        resolution,
        image_path: str,
        depth_cam_path: str | None,
        depth_est_path: str | None,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R # W2C.T
        self.T = T # W2C
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.FoVx_old = FoVx
        self.FoVy_old = FoVy

        self.image_name = image_name

        self.resolution = resolution
        self.resolution_original = resolution
        # self.data_device = torch.device("cpu")

        self.__original_image = image_path
        # move to device at dataloader to reduce VRAM requirement
        self.__sensor_depth = depth_cam_path + ".png" if depth_cam_path is not None else None
        self.__pred_depth = depth_est_path + ".png" if depth_est_path is not None else None

        # self.image_width = resolution[0]
        # self.image_height = resolution[1]

        self._gt = load_image(resolution, image_path, depth_cam_path, depth_est_path) if Camera.preload else None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def gt(self, release=True):
        if self._gt is None:
            original_image, gt_alpha_mask, sensor_depth, pred_depth = load_image(
                self.resolution_original,
                self.__original_image,
                self.__sensor_depth,
                self.__pred_depth,
            )
            gt = GroundTruth(
                original_image,
                gt_alpha_mask,
                sensor_depth,
                pred_depth,
            )
        else: 
            gt = self._gt

        if release:
            self._gt = None
        else:
            self._gt = gt

        return gt

    def modify_extrinsics(self, R_C2W, T_W2C):
        self.R = R_C2W
        self.T = T_W2C

        self.world_view_transform = (
            torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale))
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def scale_resolution(self, scale:float):
        self.__update_resolution(
            (int(self.resolution[0] * scale), int(self.resolution[1] * scale))
        )

    def __update_resolution(self, resolution):
        self.resolution = resolution

    def restore_resolution(self):
        self.resolution = self.resolution_original

    def scale_fov(self, scale_x, scale_y):
        self.__update_fov(self.FoVx * scale_x, self.FoVy * scale_y)

    def __update_fov(self, fovx, fovy):
        self.FoVx = fovx
        self.FoVy = fovy
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    def restore_fov(self):
        self.__update_fov(self.FoVx_old, self.FoVy_old)

class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def load_image(
    resolution,
    image_path: str,
    depth_cam_path: str | None = None,
    depth_est_path: str | None = None,
):
    """
    Load the image and depth maps from the FreeImage objects
    RAM usage by PIL would be free
    return Tensor are all in device cpu
    """
    image_pil = Image.open(image_path)
    depth_cam_pil = Image.open(depth_cam_path) if depth_cam_path is not None else None
    depth_est_pil = Image.open(depth_est_path) if depth_est_path is not None else None

    if len(image_pil.split()) > 3:
        # assert False, "Image has more than 3 channels, not supported"
        import torch

        resized_image_rgb = torch.cat(
            [PILtoTorch(im, resolution) for im in image_pil.split()[:3]], dim=0
        )
        loaded_mask = PILtoTorch(image_pil.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(image_pil, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    resized_depth_cam = PILtoTorch(depth_cam_pil, resolution, scale=1e3) if depth_cam_pil is not None else None
    resized_depth_est = PILtoTorch(depth_est_pil, resolution, scale=1e3) if depth_est_pil is not None else None

    image_pil.close()
    if depth_cam_pil is not None:
        depth_cam_pil.close()
    if depth_est_pil is not None:
        depth_est_pil.close()

    return GroundTruth(
        gt_image.clamp(0.0, 1.0), loaded_mask, resized_depth_cam, resized_depth_est
    )
