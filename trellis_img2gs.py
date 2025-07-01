import os
from typing import Literal

import torch
from tqdm import tqdm

os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

from PIL import Image
from submodules.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from submodules.TRELLIS.trellis.utils import render_utils, postprocessing_utils
import argparse

@torch.no_grad()
def generate(
    pipeline: TrellisImageTo3DPipeline,
    rgba_dir: str,
    object_name: str,
    save_dir: str,
    save_type: Literal["gs", "mesh", "both"] = "gs",
): 
    rgba_object_dir = os.path.join(rgba_dir, object_name)
    if not os.path.exists(rgba_object_dir):
        raise ValueError(f"RGBA object directory {rgba_object_dir} does not exist")

    # Load an image
    img_files = os.listdir(rgba_object_dir)
    if len(img_files) == 1:  # single image to 3d
        print("Processing single image")
        img_path = os.path.join(rgba_object_dir, img_files[0])
        image = Image.open(img_path)
        # Run the pipeline
        outputs = pipeline.run(image)
    else:
        # Load an image
        print("Processing multiple images")
        images = [Image.open(os.path.join(rgba_object_dir, img_file)) for img_file in img_files]
        # Run the pipeline
        outputs = pipeline.run_multi_image(
            images,
            seed=1,
            # Optional parameters
            sparse_structure_sampler_params={
                "steps": 12,
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": 12,
                "cfg_strength": 3,
            },
        )
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes
    gs_path = os.path.join(save_dir, f"{object_name}.ply")
    if save_type == "gs":
        outputs['gaussian'][0].save_ply(gs_path)
    elif save_type == "mesh":
        glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
        )
        glb.export(os.path.join(save_dir, f"{object_name}.glb"))
    elif save_type == "both":
        outputs['gaussian'][0].save_ply(gs_path)

        glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
        )
        glb.export(os.path.join(save_dir, "mesh.glb"))
    else:
        raise ValueError(f"Invalid save_type {save_type}. Must be one of ['gs', 'mesh', 'both']")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--save_type", type=str, default="gs", help="gs, mesh, both")
    parser.add_argument('--object_name', type=str, default=None)
    args = parser.parse_args()

    # Load a pipeline from a model folder or a Hugging Face model hub.
    pipeline = TrellisImageTo3DPipeline.from_pretrained("checkpoints/TRELLIS/TRELLIS-image-large")
    pipeline.cuda()

    rgba_path = os.path.join(args.model_path, "masked_image_rgba_selected")
    assert os.path.exists(rgba_path), f"RGBA path {rgba_path} does not exist"

    save_dir = os.path.join(args.model_path, "generated")
    os.makedirs(save_dir, exist_ok=True)

    if args.object_name is not None:
        # Generate a single object

        if args.object_name not in os.listdir(rgba_path):
            raise ValueError(f"Object {args.object_name} not found in {rgba_path}")
        print(f"Processing {args.object_name}")
        generate(
            pipeline,
            rgba_path,
            args.object_name,
            save_dir,
            save_type=args.save_type,
        )
        return

    # Generate 3D assets
    for object_name in tqdm(os.listdir(rgba_path)):
        print(f"Processing {object_name}")
        generate(
            pipeline,
            rgba_path,
            object_name,
            save_dir,
            save_type=args.save_type,
        )

if __name__ == '__main__':
    main()
