from argparse import ArgumentParser
import os
import sys

import torch
import yaml
from gs3dgs.arguments import ModelParams, PipelineParams, get_combined_args
from gs3dgs.gaussian_renderer import render
from gs3dgs.scene import Scene
from gs3dgs.scene.cameras import Camera
from gs3dgs.scene.gaussian_model import GaussianModel
from gs3dgs.utils.general_utils import safe_state
from utils.mask import get_mask3d


def render_func(pipeline, background):
    return lambda camera, gs, color: render(
        camera, gs, pipeline, background, override_color=color
    )

@torch.no_grad()
def gs_clone(gaussians: GaussianModel, gaussians_clone: GaussianModel):
    assert (
        gaussians.max_sh_degree == gaussians_clone.max_sh_degree
    ), "Gaussian models have different SH degrees"
    gaussians_clone.active_sh_degree = gaussians.active_sh_degree
    gaussians_clone._xyz = gaussians._xyz.clone()
    gaussians_clone._features_dc = gaussians._features_dc.clone()
    gaussians_clone._features_rest = gaussians._features_rest.clone()
    gaussians_clone._scaling = gaussians._scaling.clone()
    gaussians_clone._rotation = gaussians._rotation.clone()
    gaussians_clone._opacity = gaussians._opacity.clone()
    gaussians_clone.max_radii2D = torch.zeros(gaussians._xyz.shape[0], device="cuda")


@torch.no_grad()
def apply_mask3d(gaussians: GaussianModel, mask3d, masked_gs_path: str, return_clone_gs=False):

    gaussians_clone = GaussianModel(gaussians.max_sh_degree)
    gs_clone(gaussians, gaussians_clone)
    gaussians_clone._xyz = gaussians_clone._xyz[mask3d]
    gaussians_clone._features_dc = gaussians_clone._features_dc[mask3d]
    gaussians_clone._features_rest = gaussians_clone._features_rest[mask3d]
    gaussians_clone._scaling = gaussians_clone._scaling[mask3d]
    gaussians_clone._rotation = gaussians_clone._rotation[mask3d]
    gaussians_clone._opacity = gaussians_clone._opacity[mask3d]
    gaussians_clone.max_radii2D = gaussians_clone.max_radii2D[mask3d]
    gaussians_clone.save_ply(masked_gs_path)

    if return_clone_gs:
        return gaussians_clone

    # free VRAM in 'gaussians_clone'
    del gaussians_clone
    return None

def mask3d(dataset: ModelParams, pipeline: PipelineParams):

    description_path = os.path.join(dataset.source_path, "description.yml")
    if not os.path.exists(description_path):
        raise FileNotFoundError(f"Config file not found at {description_path}")
    
    with open(description_path, "r") as f:
        description = yaml.safe_load(f)
    
    # iteration = description["iteration"] if "iteration" in description else 25000
    iteration = 7000
    prompts = description["prompts"]
    prompts = [prompt.strip() for prompt in prompts.split(",")]
    ext = description["ext"]

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_f = render_func(pipeline, background)

    masks_3d = get_mask3d(
        render_function=render_f,
        gaussians=gaussians,
        cam_list_train=scene.getTrainCameras().copy(),
        cam_list_test=scene.getTestCameras().copy(),
        prompts=prompts,
        data_dir=dataset.source_path,
        ext=ext,
    )
    with torch.no_grad():
        masks_3d_inverted = (masks_3d.sum(dim=0) > 0).logical_not()
    gs_seg_path = os.path.join(dataset.model_path, "gs_seg")
    os.makedirs(gs_seg_path, exist_ok=True)
    for mask_3d, prompt in zip(masks_3d, prompts):
        apply_mask3d(gaussians, mask_3d, os.path.join(gs_seg_path, f"{prompt}.ply"))
    gs_remained = apply_mask3d(gaussians, masks_3d_inverted, os.path.join(gs_seg_path, f"remained.ply"), return_clone_gs=True)


def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    safe_state(False)
    mask3d(model.extract(args), pipeline.extract(args))

if __name__ == "__main__":
    main()
