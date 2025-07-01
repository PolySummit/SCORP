import math
import os
import shutil
import numpy as np
import torch
from torchvision.ops import box_convert

from tqdm import tqdm

import cv2
from typing import Callable, Literal
from sam2.build_sam import build_sam2_video_predictor

from gs2dgs.scene.cameras import Camera as gs2dgsCamera
from gs3dgs.scene.cameras import Camera as gs3dgsCamera
from gs2dgs.scene.gaussian_model import GaussianModel as gs2dgsGaussianModel
from gs3dgs.scene.gaussian_model import GaussianModel as gs3dgsGaussianModel

# import pdb

def _camera_intrinsic_transform(fov_x, fov_y, resolution):
    pixel_width, pixel_height = resolution
    camera_intrinsics = np.zeros((3, 3))
    camera_intrinsics[2, 2] = 1
    camera_intrinsics[0, 0] = (pixel_width / 2.0) / math.tan(math.radians(fov_x / 2.0))
    camera_intrinsics[0, 2] = pixel_width / 2.0
    camera_intrinsics[1, 1] = (pixel_height / 2.0) / math.tan(math.radians(fov_y / 2.0))
    camera_intrinsics[1, 2] = pixel_height / 2.0
    return camera_intrinsics

def get_mask3d(
    render_function: Callable[..., dict],
    gaussians: gs2dgsGaussianModel | gs3dgsGaussianModel,
    cam_list_train: list[gs2dgsCamera | gs3dgsCamera],
    cam_list_test: list[gs2dgsCamera | gs3dgsCamera],
    prompts: list[str],
    data_dir: str,
    voting_method: Literal["gradient", "binary", "projection"] = "gradient",
    ext: str = "png",
):

    @torch.enable_grad()
    def _iter_one_frame():
        image_name = camera.image_name
        width, height = camera.resolution

        colors = torch.ones((gaussians.get_xyz.shape[0], 3), requires_grad=True, device="cuda")
        rendered_pkg = render_function(camera, gaussians, colors)
        rendered_color: torch.Tensor = rendered_pkg["render"]
        meta = None
        target = rendered_color.permute(1, 2, 0).mean()
        target.backward(retain_graph=True)

        masks = [
            cv2.imread(os.path.join(mask_with_image_prompt_rgba_dir, f"{image_name}.png"), cv2.IMREAD_UNCHANGED)[:, :, 3] > 0
            for mask_with_image_prompt_rgba_dir in mask_with_image_prompt_rgba_dir_list
        ]

        for object_id, mask_org in zip(range(len(prompts)), masks):
            # mask = (cv2.erode(mask_org.astype(float), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0).astype(bool)
            mask = mask_org.astype(bool)
            target = rendered_color.permute(1, 2, 0) * torch.from_numpy(mask).to("cuda", non_blocking=True)[..., None]
            colors.grad.zero_()
            loss = 1 * target.mean()
            loss.backward(retain_graph=True)
            if voting_method == "gradient":
                # mins = torch.min(colors.grad, dim=-1).values
                # maxes = torch.max(colors.grad, dim=-1).values
                # assert torch.allclose(mins , maxes), f"Something is wrong with gradient calculation {mins} {maxes}"
                gaussian_grads[object_id] += (colors.grad).norm(dim=[1])

            elif voting_method == "binary":
                gaussian_grads[object_id] += 1 * (colors.grad.norm(dim=[1]) > 0)
            elif voting_method == "projection":
                means2d = np.round(meta["means2d"].detach().cpu().numpy()).astype(int)
                means2d_mask = (means2d[:, 0] >= 0) & (means2d[:, 0] < width) & (means2d[:, 1] >= 0) & (means2d[:, 1] < height)
                means2d = means2d[means2d_mask]
                gaussian_ids = meta["gaussian_ids"].detach().cpu().numpy()
                gaussian_ids = gaussian_ids[means2d_mask]
                means2d_mask = mask[means2d[:, 1], means2d[:, 0]] # Check if the splat is in the mask
                gaussian_grads[object_id][torch.from_numpy(gaussian_ids[~means2d_mask]).long()] -= 1
                gaussian_grads[object_id][torch.from_numpy(gaussian_ids[means2d_mask]).long()] += 1
            else:
                raise ValueError("Invalid voting method")
            mask_inverted = ~mask
            target = rendered_color.permute(1, 2, 0) * torch.from_numpy(mask_inverted).to("cuda",non_blocking=True)[..., None]
            colors.grad.zero_()
            loss = 1 * target.mean()
            loss.backward(retain_graph=True)
            if voting_method == "gradient":
                gaussian_grads[object_id] -= (colors.grad).norm(dim=[1])
            elif voting_method == "binary":
                gaussian_grads[object_id] -= 1 * ((colors.grad).norm(dim=[1]) > 0)
            elif voting_method == "projection":
                pass
            else:
                raise ValueError("Invalid voting method")
            colors.grad.zero_()

        return

    with torch.no_grad():
        gaussian_grads = torch.zeros(len(prompts), gaussians.get_xyz.shape[0], device="cuda")

    mask_with_image_rgba_dir = os.path.join(data_dir, "masked_image_rgba")
    if not os.path.exists(mask_with_image_rgba_dir):
        raise RuntimeError(f"`mask_with_image_rgba_dir` {mask_with_image_rgba_dir} does not exist")

    mask_with_image_prompt_rgba_dir_list = [os.path.join(mask_with_image_rgba_dir, prompt) for prompt in prompts]

    for mask_with_image_prompt_rgba_dir in mask_with_image_prompt_rgba_dir_list:
        if not os.path.exists(mask_with_image_prompt_rgba_dir):
            raise RuntimeError(f"`mask_with_image_prompt_rgba_dir` {mask_with_image_prompt_rgba_dir} does not exist")

    # os.makedirs(f"{mask_with_image_dir}/masked", exist_ok=True)
    os.makedirs(f"{mask_with_image_rgba_dir}/masked", exist_ok=True)
    # os.makedirs(f"{mask_bin_dir}/masked", exist_ok=True)
    # propagate the prompts to get masklets throughout the video
    # frame_idx = 0
    with torch.no_grad():
        for camera in tqdm(cam_list_train):
            _iter_one_frame()

    masks_3d = gaussian_grads > 0
    # mask_3d_inverted = gaussian_grads <= 0 # We don't need Gaussians without any influence ie gaussian_grads == 0
    return masks_3d