from argparse import ArgumentParser
import os
import shutil
import sys

import torch
import numpy as np
from tqdm import tqdm

from gs3dgs.arguments import ModelParams, PipelineParams, get_combined_args
from gs3dgs.gaussian_renderer import render
from gs3dgs.scene import Scene
from gs3dgs.scene.gaussian_model import GaussianModel
from gs3dgs.utils.general_utils import safe_state, TorchtoPIL


@torch.no_grad
def _render_images(
    dataset: ModelParams,
    pipeline: PipelineParams,
    pretrained_ply_list: list[str],
    render_save_dir: str,
):
    # render_save_dir = os.path.join(dataset.model_path, "rendered_refined")
    shutil.rmtree(render_save_dir, ignore_errors=True)
    os.makedirs(render_save_dir)

    gaussians = GaussianModel(dataset.sh_degree)
    # pretrained_ply_list = [
    #     # os.path.join(dataset.model_path, "gs_seg", "remained.ply"),
    #     os.path.join(dataset.model_path, "refined_aligned", f"refined_{iteration}.ply"),
    # ]
    scene = Scene(
        dataset, gaussians, shuffle=False, pretrained_ply_path_list=pretrained_ply_list
    )

    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    bg_color = [1, 1, 1]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"{len(gaussians._xyz)} Gaussians loaded")

    for viewpoint_cam in tqdm(scene.getTestCameras()):
        render_pkg = render(viewpoint_cam, gaussians, pipeline, bg)
        rendered_image: torch.Tensor = render_pkg["render"]
        rendered_alpha: torch.Tensor = (render_pkg["render_alpha"] > 0).to(
            torch.float32
        )

        rendered_rgba = torch.cat((rendered_image, rendered_alpha), dim=0)
        rendered_rgba_pil = TorchtoPIL(rendered_rgba)

        rendered_rgba_pil.save(
            os.path.join(render_save_dir, f"{viewpoint_cam.image_name}.png")
        )


@torch.no_grad()
def render_images(
    dataset: ModelParams,
    pipeline: PipelineParams,
    iteration: int = 1000,
    render_proposed: bool = True,
    render_baseline: bool = True,
    render_refined_dir_name = "refined_aligned",
):
    if render_proposed:
        _render_images(
            dataset,
            pipeline,
            pretrained_ply_list=[
                os.path.join(
                    dataset.model_path, "refined_aligned", f"refined_{iteration}.ply"
                ),
            ],
            render_save_dir=os.path.join(
                dataset.model_path,
                render_refined_dir_name,
            ),
        )

    if render_baseline:
        _render_images(
            dataset,
            pipeline,
            pretrained_ply_list=[
                os.path.join(dataset.model_path, "gs_seg", gs_name)
                for gs_name in os.listdir(os.path.join(dataset.model_path, "gs_seg"))
                if (gs_name.endswith(".ply") and gs_name != "remained.ply")
            ],
            render_save_dir=os.path.join(
                dataset.model_path,
                "rendered_baseline",
            ),
        )


def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--iter", type=int, default=1000)
    parser.add_argument("--not_render_proposed", action="store_true", default=False)
    parser.add_argument("--not_render_baseline", action="store_true", default=False)
    parser.add_argument(
        "--render_refined_dir_name", type=str, default="rendered_refined"
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    args = get_combined_args(parser)

    safe_state(False)
    render_images(
        model.extract(args),
        pipeline.extract(args),
        iteration=args.iter,
        render_proposed=not args.not_render_proposed,
        render_baseline=not args.not_render_baseline,
        render_refined_dir_name=args.render_refined_dir_name,
    )


if __name__ == "__main__":
    main()
