# raise ImportError("Temporary error, please ignore this.")

from argparse import ArgumentParser
# import colorsys
import os
import random
import shutil
import sys

import numpy as np
import torch
from tqdm import tqdm
import yaml

from utils.graphic import get_centroid, pix2pcd
from utils.match import get_pairwise_mask3r_features as get_pairwise_features
from utils.image import crop_with_alpha, restore_coords, show_feature_matches
from utils.solution import adam_algorithm_3d3d_9dof, pc_align_ransac
from gs3dgs.arguments import ModelParams, PipelineParams, get_combined_args
from gs3dgs.gaussian_renderer import render as render_3dgs
from gs3dgs.scene import Scene
from gs3dgs.scene.cameras import Camera
from gs3dgs.scene.gaussian_model import GaussianModel
from gs3dgs.utils.general_utils import TorchtoPIL, safe_state, PILtoTorch
from gs3dgs.utils.graphics_utils import fov2focal

from PIL import Image

import open3d as o3d

from utils.gaussians import gaussians_rotate, gaussians_scale, gaussians_translate

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def render_func_3dgs(pipeline, background):
    return lambda camera, gs, color: render_3dgs(
        camera, gs, pipeline, background, override_color=color
    )

def get_ICP_fitting_transformation_best(
    pc_xyz_original: np.ndarray,
    pc_xyz_refined: np.ndarray,
    rotations: np.ndarray,
    threshold: float,
) -> np.ndarray:

    if np.any(np.isnan(pc_xyz_original)) or np.any(np.isnan(pc_xyz_refined)):
        raise ValueError("Point clouds contain NaN values")
    if np.any(np.isinf(pc_xyz_original)) or np.any(np.isinf(pc_xyz_refined)):
        raise ValueError("Point clouds contain Inf values")

    # pc_xyz_original_avg = pc_xyz_original.mean(axis=0)
    # pc_xyz_refined_avg = pc_xyz_refined.mean(axis=0)

    center_original = get_centroid(pc_xyz_original, method="mean")
    center_refined = get_centroid(pc_xyz_refined, method="mean")

    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(pc_xyz_original)
    pcd_refined = o3d.geometry.PointCloud()
    pcd_refined.points = o3d.utility.Vector3dVector(pc_xyz_refined)

    # downsample the pcd_refined to speed up ICP. num point should be close to the num of pcd_original
    num_points_original = len(pc_xyz_original)
    num_points_refined = len(pc_xyz_refined)

    if num_points_refined > 4 * num_points_original:
        every_k_points = int(num_points_refined / (4 * num_points_original))
        pcd_refined = pcd_refined.uniform_down_sample(every_k_points)
        print("ICP downsampled point cloud size:", len(pcd_refined.points))
        print("ICP original point cloud size:", len(pcd_original.points))

    print("ICP threshold:", threshold)
    print("rotation matrices initializing...")

    # Generate mutiple transformation matrices with different initializations on rotation

    trans_inits = (len(rotations) + 3) * [None]
    for idx, rot in enumerate(rotations):
        trans_init = np.eye(4)
        trans_init[:3, :3] = rot
        trans_init[:3, 3] = center_original - rot @ center_refined
        trans_inits[idx] = trans_init

    trans_inits[-3] = np.eye(4)
    trans_inits[-3][:3, 3] = center_original - center_refined
    trans_inits[-2] = np.eye(4)
    trans_inits[-2][:3, 3] = pc_xyz_original.mean(axis=0) - pc_xyz_refined.mean(axis=0)
    trans_inits[-1] = np.eye(4)

    print("Running ICP...")

    best_fitness = -np.inf
    best_transform = None

    for trans_init in tqdm(trans_inits):

        result = o3d.pipelines.registration.registration_icp(
            pcd_refined, pcd_original, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=400
            )
        )

        # Update best result if current is better
        if result.fitness > best_fitness:
            best_fitness = result.fitness
            best_transform = result.transformation

    print("Best ICP transformation:\n", best_transform)
    print("Best fitness:", best_fitness)
    return best_transform

@torch.no_grad()
def get_pcd_pair(
    gaussian_original,
    gaussian_refined,
    cam_list:list[Camera],
    render_f,
    mask_images_crop: list[torch.Tensor],
    mask_depthes_crop: list[torch.Tensor],
    mask_areas: list[float],
    mask_bbox_xxyys: list[tuple[int, int, int, int]],
    iteration: int = -1,
    n_points_per_cam: int = 12,
    cam_list_interval: int = 10,
    visual_match_save_path: str = "tmp_match",
) -> tuple[np.ndarray, np.ndarray]:

    device = torch.device("cuda")
    start = iteration % cam_list_interval

    __tmp_cam = cam_list[0]
    w_original, h_original = __tmp_cam.resolution
    fx_original = fov2focal(__tmp_cam.FoVx, w_original)
    fy_original = fov2focal(__tmp_cam.FoVy, h_original)
    cx_original = w_original // 2
    cy_original = h_original // 2

    points_original = []
    points_refined = []

    for idx in range(start, len(cam_list), cam_list_interval):
        cam = cam_list[idx]
        _depth_original_rendered_crop = mask_depthes_crop[idx]
        _area_original = mask_areas[idx]
        if _area_original <= 0:
            print(f"Camera {idx} has no area on the input image, skipping...")
            continue

        xxyy_original = mask_bbox_xxyys[idx]

        # scale up the resolution to get more points
        for _scale_idx in range(4):
            if _scale_idx > 0:
                cam.scale_resolution(1.5)
            _render_pkg = render_f(cam, gaussian_refined, None)
            _image_refined_rendered: torch.Tensor = _render_pkg["render"].cpu()
            _depth_refined_rendered: torch.Tensor = _render_pkg["render_depth"].cpu()
            _mask_refiend_rendered: torch.Tensor = _render_pkg["render_alpha"].cpu()
            del _render_pkg
            _mask_refiend_rendered = _mask_refiend_rendered > 0.0
            _area_refined_rendered = torch.sum(_mask_refiend_rendered).item()

            if _area_refined_rendered > _area_original / 10:
                break

        if _mask_refiend_rendered.sum() <= 0:
            print(f"Camera {idx} has no area on the rendered image, skipping...")
            cam.restore_resolution()
            continue

        _depth_refined_rendered = (_depth_refined_rendered * _mask_refiend_rendered)
        _image_refined_rendered_crop, _depth_refined_rendered_crop, xxyy_refined = (
            crop_with_alpha(
                _image_refined_rendered,
                _mask_refiend_rendered,
                _depth_refined_rendered,
                border=200,
            )
        )

        # image pil prepare
        _image_original_crop_pil = TorchtoPIL(mask_images_crop[idx])
        _image_refined_rendered_crop_pil = TorchtoPIL(_image_refined_rendered_crop)

        _depth_original_rendered_crop_np = _depth_original_rendered_crop.squeeze().cpu().numpy() # [H, W]
        _depth_refined_rendered_crop_np = _depth_refined_rendered_crop.squeeze().cpu().numpy()

        viz_matches_im0_, viz_matches_im1_ = get_pairwise_features(
            _image_original_crop_pil,
            _image_refined_rendered_crop_pil,
            n_points_per_cam=n_points_per_cam,
        ) # [N, 2] # num_matches, (u, v) 

        # get points pair
        depths_original = _depth_original_rendered_crop_np[viz_matches_im0_[:, 1], viz_matches_im0_[:, 0]]
        depths_refined = _depth_refined_rendered_crop_np[viz_matches_im1_[:, 1], viz_matches_im1_[:, 0]]
        depths_nonzero_idx = (depths_original > 0) & (depths_refined > 0)

        # filter out zero depth points
        depths_original = depths_original[depths_nonzero_idx]
        viz_matches_im0_ = viz_matches_im0_[depths_nonzero_idx]
        depths_refined = depths_refined[depths_nonzero_idx]
        viz_matches_im1_ = viz_matches_im1_[depths_nonzero_idx]

        w_refined, h_refined = cam.resolution
        fx_refined = fov2focal(cam.FoVx, w_refined)
        fy_refined = fov2focal(cam.FoVy, h_refined)
        cx_refined = w_refined // 2
        cy_refined = h_refined // 2

        viz_matches_im0_restored = restore_coords(viz_matches_im0_, xxyy_original)
        viz_matches_im1_restored = restore_coords(viz_matches_im1_, xxyy_refined)

        points_original_cam = pix2pcd(
           viz_matches_im0_restored,
            depths_original,
            fx_original,
            fy_original,
            cx_original,
            cy_original,
        )

        points_refined_cam = pix2pcd(
            viz_matches_im1_restored,
            depths_refined,
            fx_refined,
            fy_refined,
            cx_refined,
            cy_refined,
        )

        R_c2w: np.ndarray = cam.R
        t_c2w: np.ndarray = -R_c2w @ cam.T

        points_original += [points_original_cam @ R_c2w.T + t_c2w]
        points_refined += [points_refined_cam @ R_c2w.T + t_c2w]

        if visual_match_save_path:
            show_feature_matches(
                viz_matches_im0_,
                viz_matches_im1_,
                _image_original_crop_pil,
                _image_refined_rendered_crop_pil,
                os.path.join(visual_match_save_path, f"{iteration}_{idx}.png")
            )

        # reset resolution
        print(f"{idx}th camera done. {len(depths_nonzero_idx)} points found.", end="\r")
        cam.restore_resolution()

    points_original = np.concatenate(points_original)
    points_refined = np.concatenate(points_refined)

    return points_original, points_refined 


@torch.no_grad()
def apply_scale(gaussian_refined, scale):
    scale = np.array(scale, dtype=float)
    if np.isnan(scale).any():
        raise ValueError("Scale cannot be Nan")
    if scale.size == 1:
        scale_xyz = torch.tensor(np.stack([scale, scale, scale]), dtype=torch.float32, device="cuda")
        gaussians_scale(gaussian_refined, scale_xyz)
    elif scale.size == 3:
        scale_xyz = torch.tensor(scale, dtype=torch.float32, device="cuda")
        gaussians_scale(gaussian_refined, scale_xyz)
    else:
        raise ValueError("Scale must be a scalar or a 3-element array")
    
    print(f"scaled by {scale}")

@torch.no_grad()
def apply_transformation(gaussian_refined, R, t):
    t = torch.tensor(t, dtype=torch.float32, device="cuda")
    R = torch.tensor(R, dtype=torch.float32, device="cuda")
    gaussians_rotate(gaussian_refined, R)
    gaussians_translate(gaussian_refined, t)

def align(
    dataset: ModelParams,
    pipeline: PipelineParams,
    object_name: str,
    num_iterations: int = 6,
    opt_12dof_iterations: list = [3],
    cam_list_interval: int = 10,
    rotations_path: str = "rotation_matrices",
    visual_match: bool = False,
):
    print(pipeline.__dict__)

    rotations_list = (
        np.load(os.path.join(rotations_path, "rotations_64.npz"))["rotations"]
        if rotations_path
        else None
    )

    with torch.no_grad():
        scene = Scene(dataset, None, shuffle=False)
        gaussian_original = GaussianModel(dataset.sh_degree)
        gaussian_original.load_ply(os.path.join(dataset.model_path, "gs_seg", f"{object_name}.ply"))
        gaussian_refined = GaussianModel(0)
        gaussian_refined.load_ply(os.path.join(dataset.model_path, "generated", f"{object_name}.ply"))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_f_3dgs = render_func_3dgs(pipeline, background)

    # print("Loading rotations...")
    # assert os.path.exists(rotations_path), f"Rotations file {rotations_path} does not exist"
    # rotations = np.load(rotations_path)["rotations"]

    cam_list = scene.getTrainCameras().copy()
    rgba_dir = os.path.join(dataset.source_path, "masked_image_rgba")
    assert os.path.exists(rgba_dir), f"RGBA directory {rgba_dir} does not exist"

    # cam_list_interval = 1
    cam_list_interval = np.ceil(len(cam_list) / 15).astype(int)
    # cam_list_interval = np.ceil(len(cam_list) / 30).astype(int)
    # print(np.ceil(len(cam_list) / 15))
    # print(cam_list_interval)
    # assert False

    print("Loading RGBA...")
    original_mask_areas: list[float] = []
    original_mask_images: list[torch.Tensor] = []
    original_mask_depthes: list[torch.Tensor] = []
    original_xxyys: list[tuple[int, int, int, int]] = []
    with torch.no_grad():
        for cam in tqdm(cam_list):
            _rgba_path = os.path.join(rgba_dir, object_name, f"{cam.image_name}.png")
            _rgba = Image.open(_rgba_path)
            # _mask = _mask.filter(ImageFilter.MaxFilter(11))
            _rgba = PILtoTorch(_rgba, cam.resolution, scale=255.0)
            _mask = (_rgba[3, :, :] > 0).unsqueeze(0)
            _mask_area = torch.sum(_mask).item()

            if _mask_area > 0:
                _image = _rgba[:3, :, :]
                _image = _image * _mask

                _render_pkg = render_f_3dgs(cam, gaussian_original, None)
                _depth: torch.Tensor = _render_pkg["render_depth"].cpu()
                del _render_pkg
                _depth = _depth * _mask

                _image_crop, _depth_crop, xxyy = crop_with_alpha(
                    _image,
                    _mask,
                    _depth,
                    border=200,
                )

                original_xxyys.append(xxyy)
                original_mask_areas.append(_mask_area)
                original_mask_images.append(_image_crop)
                original_mask_depthes.append(_depth_crop)
            else:
                original_xxyys.append(None)
                original_mask_areas.append(0.0)
                original_mask_images.append(None)
                original_mask_depthes.append(None)

    aligned_output_dir = os.path.join(dataset.model_path, "generated_aligned")
    os.makedirs(aligned_output_dir, exist_ok=True)

    with torch.no_grad():
        pc_xyz_original = gaussian_original._xyz.cpu().numpy().astype(np.float32)
        pc_xyz_refined = gaussian_refined._xyz.cpu().numpy().astype(np.float32)

    bbox_size_original = np.max(pc_xyz_original, axis=0) - np.min(pc_xyz_original, axis=0)
    bbox_size_refined = np.max(pc_xyz_refined, axis=0) - np.min(pc_xyz_refined, axis=0)
    bbox_size_ratio = bbox_size_original / bbox_size_refined
    scale_bbox = np.prod(bbox_size_ratio) ** (1 / 3)

    threshold_matching = bbox_size_original.mean() / 10

    apply_scale(gaussian_refined, scale_bbox)

    # if rotations_list is not None:
    #     rotation = random.choice(rotations_list)
    #     apply_transformation(gaussian_refined, rotation, [0.0, 0.0, 0.0])

    translation_pre = get_centroid(pc_xyz_original,"mean") - get_centroid(pc_xyz_refined,"mean")
    apply_transformation(gaussian_refined, np.eye(3), translation_pre)

    if rotations_list is not None:
        with torch.no_grad():
            pc_xyz_refined = gaussian_refined._xyz.cpu().numpy().astype(np.float32)

        best_transform_icp = get_ICP_fitting_transformation_best(
            pc_xyz_original,
            pc_xyz_refined,
            rotations_list,
            threshold=threshold_matching * 1.6,
        )
        best_t_icp = best_transform_icp[:3, 3]
        best_R_icp = best_transform_icp[:3, :3]
        apply_transformation(gaussian_refined, best_R_icp, best_t_icp)

    # shutil.rmtree(visual_match_save_path, ignore_errors=True)
    if visual_match:
        visual_match_path = os.path.join(dataset.model_path, "visual_match")
        os.makedirs(visual_match_path, exist_ok=True)
        visual_match_save_object_path = os.path.join(visual_match_path, object_name)
        shutil.rmtree(visual_match_save_object_path, ignore_errors=True)
        os.makedirs(visual_match_save_object_path)

    for _iter in tqdm(range(num_iterations), desc="Optimizating..."):
        # get corresponding pcd pair
        original_gen_pcd, refined_gen_pcd = get_pcd_pair(
            gaussian_original,
            gaussian_refined,
            cam_list,
            render_f_3dgs,
            original_mask_images,
            original_mask_depthes,
            original_mask_areas,
            original_xxyys,
            _iter,
            n_points_per_cam=16,
            # n_points_per_cam=12,
            cam_list_interval=cam_list_interval,
            visual_match_save_path=visual_match_save_object_path,
        )

        R_org = None

        # get scale and transformation
        if _iter in opt_12dof_iterations:
            R, t, s, R_org = adam_algorithm_3d3d_9dof(
                refined_gen_pcd,
                original_gen_pcd,
                iterations=3000,
                verbose_interval=200,
            )
            # R, t, s = pc_align_ransac(
            #     refined_gen_pcd,
            #     original_gen_pcd,
            #     threshold=threshold_matching,
            #     method="umeyama_gen",
            # )
        else:
            R, t, s = pc_align_ransac(
                refined_gen_pcd,
                original_gen_pcd,
                threshold=threshold_matching,
            )

        if R_org is not None:
            apply_transformation(gaussian_refined, R_org, [0.0, 0.0, 0.0])

        # apply the scale to the refined object
        apply_scale(gaussian_refined, s)

        if R_org is not None:
            apply_transformation(gaussian_refined, R_org.T, [0.0, 0.0, 0.0])

        # apply the transformation to the refined object
        apply_transformation(gaussian_refined, R, t)

        gaussian_refined_save_path = os.path.join(aligned_output_dir, f"{object_name}_{_iter}.ply")
        gaussian_refined.save_ply(gaussian_refined_save_path)
        print(f"Aligned object saved to {gaussian_refined_save_path}")

    gaussian_refined_save_path = os.path.join(aligned_output_dir, f"{object_name}.ply")
    gaussian_refined.save_ply(gaussian_refined_save_path)
    print(f"Aligned object saved to {gaussian_refined_save_path}")

def align_objects(
    dataset: ModelParams,
    pipeline: PipelineParams,
    num_iterations: int = 6,
    opt_12dof_iterations: list = [3],
    cam_list_interval: int = 10,
    rotations_path: str = "rotation_matrices",
    visual_match: bool = False,
):
    rgba_selected_path = os.path.join(dataset.model_path, "masked_image_rgba_selected")
    if not os.path.exists(rgba_selected_path):
        raise FileNotFoundError(f"RGBA path not found at {rgba_selected_path}")
    object_names = os.listdir(rgba_selected_path)
    for object_name in tqdm(object_names):
        align(
            dataset,
            pipeline,
            object_name,
            num_iterations,
            opt_12dof_iterations,
            cam_list_interval,
            rotations_path,
            visual_match=visual_match,
        )

def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--name", type=str, default="None", help="name of the object to replace")
    parser.add_argument("--rotations_dir", default="rotation_matrices", type=str, help="path to the rotations file")
    args = parser.parse_args(sys.argv[1:])

    dataset = model.extract(args)

    description_path = os.path.join(dataset.source_path, "description.yml")
    if not os.path.exists(description_path):
        raise FileNotFoundError(f"Config file not found at {description_path}")

    with open(description_path, "r") as f:
        description = yaml.safe_load(f)

    num_iterations = description["num_iterations"] if "num_iterations" in description else 6
    # num_iterations = 6
    opt_12dof_iterations = description["opt_shape_iterations"] if "opt_shape_iterations" in description else [3, 4]
    # opt_12dof_iterations = [3,4]
    cam_list_interval = description["cam_list_interval"] if "cam_list_interval" in description else 10

    safe_state(False)
    if args.name != "None":
        align(
            dataset,
            pipeline.extract(args),
            args.name,
            num_iterations,
            opt_12dof_iterations,
            cam_list_interval,
            args.rotations_dir,
            visual_match=True,
        )
    else:
        align_objects(
            dataset,
            pipeline.extract(args),
            num_iterations,
            opt_12dof_iterations,
            cam_list_interval,
            args.rotations_dir,
            visual_match=True,
        )

if __name__ == "__main__":
    main()
