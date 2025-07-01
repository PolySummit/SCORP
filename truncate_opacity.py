from argparse import ArgumentParser
import os
import shutil

import torch

from gs3dgs.scene.gaussian_model import GaussianModel

@torch.no_grad()
def truncate_gs(gs: GaussianModel, threshold: float = 0.1):

    selector = ~(gs.get_opacity < threshold).squeeze()

    gs._xyz = gs._xyz[selector]
    gs._features_dc = gs._features_dc[selector]
    gs._features_rest = gs._features_rest[selector]
    gs._scaling = gs._scaling[selector]
    gs._rotation = gs._rotation[selector]
    gs._opacity = gs._opacity[selector]
    gs.max_radii2D = gs.max_radii2D[selector]


def truncate(gs_path: str, threshold: float = 0.1):
    assert os.path.exists(gs_path), f"Gaussian model file {gs_path} does not exist"

    shutil.copy(gs_path, gs_path + ".bak")

    with torch.no_grad():
        gaussian_refined = GaussianModel(0)
        gaussian_refined.load_ply(gs_path)
        truncate_gs(gaussian_refined, threshold)

    gaussian_refined.save_ply(gs_path)


def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        required=True,
        help="path to the scene Gaussian model",
    )
    parser.add_argument(
        "--threshold",
        default=0.1,
        type=float,
        help="opacity threshold for truncation",
    )
    args = parser.parse_args()

    gs_generated_path = os.path.join(args.model_path, "generated")
    if not os.path.exists(gs_generated_path):
        raise FileNotFoundError(f"Generated Gaussian model path {gs_generated_path} does not exist")
    
    for gs_file_name in os.listdir(gs_generated_path):
        if gs_file_name.endswith(".ply"):
            gs_path = os.path.join(gs_generated_path, gs_file_name)
            truncate(gs_path, args.threshold)
            print(f"Truncated {gs_path} with threshold {args.threshold}")


if __name__ == "__main__":
    main()
