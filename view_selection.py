from argparse import ArgumentParser
import os
import shutil
import warnings

import cv2
import numpy as np
import torch
from tqdm import tqdm
import yaml

from PIL import Image

from gs3dgs.arguments import ModelParams, PipelineParams, get_combined_args
from gs3dgs.scene import Scene
from gs3dgs.scene.cameras import Camera
from gs3dgs.scene.gaussian_model import GaussianModel
from gs3dgs.utils.general_utils import TorchtoPIL, safe_state
from utils.views import (
    get_xyxy_from_mask,
    post_process_rgba_imgs,
    select_high_quality_and_diverse_images,
    select_high_quality_and_diverse_images_beta,
)
from gs3dgs.gaussian_renderer import render
from gs3dgs.utils.graphics_utils import fov2focal

def view_select(
    dataset: ModelParams,
    pipeline: PipelineParams,
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    sam2_checkpoint: str = "checkpoints/sam2.1/sam2.1_hiera_large.pt",
):
    description_path = os.path.join(dataset.source_path, "description.yml")
    if not os.path.exists(description_path):
        raise FileNotFoundError(f"Config file not found at {description_path}")

    with open(description_path, "r") as f:
        description = yaml.safe_load(f)

    ext = description["ext"]

    selected_img_num = description["img_num"] if "img_num" in description else 3

    rgba_path = os.path.join(dataset.source_path, "masked_image_rgba")
    if not os.path.exists(rgba_path):
        raise FileNotFoundError(f"RGBA path not found at {rgba_path}")

    rgba_selected_path = os.path.join(
        dataset.model_path, "masked_image_rgba_selected"
    )

    if os.path.exists(rgba_selected_path):
        shutil.rmtree(rgba_selected_path)
    os.makedirs(rgba_selected_path)

    rgba_objects = os.listdir(rgba_path)

    with torch.no_grad():
        scene = Scene(dataset, None, shuffle=False)

    train_cam_list = scene.getTrainCameras().copy()
    # test_cam_list = scene.getTestCameras().copy()

    print(f"Number of train cameras: {len(train_cam_list)}")
    # print(f"Number of test cameras: {len(test_cam_list)}")

    rotations_W2C = np.array([cam.R.T for cam in train_cam_list])
    translations_W2C = np.array([cam.T for cam in train_cam_list])
    image_names = [cam.image_name for cam in train_cam_list]

    for object in tqdm(rgba_objects):
        if object == "masked":
            continue

        rgba_object_path = os.path.join(rgba_path, object)
        if not os.path.isdir(rgba_object_path):
            raise NotADirectoryError(
                f"RGBA object path not a directory: {rgba_object_path}"
            )

        rgba_object_selected_path = os.path.join(rgba_selected_path, object)
        os.makedirs(rgba_object_selected_path, exist_ok=True)

        selected_rgba_np_list = select_high_quality_and_diverse_images_beta(
            rgba_object_path,
            image_names,
            selected_img_num,
            translations_W2C,
            R_W2C=rotations_W2C,
            quality_weight=0.25,
            mask_weight=0.25,
            diversity_weight=0.25,
        )

        size_list = [
            selected_rgba_np.shape[0] * selected_rgba_np.shape[1]
            for selected_rgba_np in selected_rgba_np_list
        ]

        max_size = np.median(size_list)
        print(size_list)
        for i, selected_rgba_size in enumerate(size_list[::-1]):
            if selected_rgba_size < max_size * 0.25:
                print(f"Removing image {len(size_list)-i-1} with size {selected_rgba_size}")
                selected_rgba_np_list.pop(len(size_list) - 1 - i)

        print(f"Selected {len(selected_rgba_np_list)} images for {object}")
        crop_rgba_imgs_pil = post_process_rgba_imgs(selected_rgba_np_list)

        # save selected images to output_dir
        for i, rgba_img_pil in enumerate(crop_rgba_imgs_pil):
            rgba_img_pil.save(os.path.join(rgba_object_selected_path, f"img_{i+1}.png"))
        print(f"Saved {len(crop_rgba_imgs_pil)} images to {rgba_object_selected_path}")


def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    args = get_combined_args(parser)

    safe_state(False)
    view_select(model.extract(args), pipeline.extract(args))

if __name__ == "__main__":
    main()
