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

import os
import shutil
import torch
from random import randint
from gs3dgs.utils.loss_utils import isotropic_loss, l1_loss, ssim
from gs3dgs.utils.image_utils import depth_normalize_, psnr
from gs3dgs.gaussian_renderer import render, network_gui
import sys
from gs3dgs.scene import Scene, GaussianModel
from gs3dgs.utils.general_utils import TorchtoPIL, get_expon_lr_func, safe_state
import uuid
from tqdm import tqdm
from gs3dgs.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from gs3dgs.arguments import ModelParams, PipelineParams, OptimizationParams
from segmentation_3dgs import apply_mask3d


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
):

    pretrained_ply_dir = os.path.join(dataset.model_path, "generated_aligned")
    if not os.path.exists(pretrained_ply_dir):
        raise FileNotFoundError(f"Pretrained PLY directory not found at {pretrained_ply_dir}")

    pretrained_ply_name_list = [ply_path for ply_path in os.listdir(os.path.join(dataset.model_path, "generated")) if ply_path.endswith(".ply")]
    pretrained_ply_path_list = [os.path.join(pretrained_ply_dir, ply_path) for ply_path in pretrained_ply_name_list]

    if len(pretrained_ply_path_list) == 0:
        raise FileNotFoundError(f"No PLY files found in the directory {pretrained_ply_dir}")

    first_iter = 0
    assert dataset.sh_degree == 0, "SH degree must be 0 for post-refine"
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, pretrained_ply_path_list = pretrained_ply_path_list)
    gs_size_list = gaussians.load_multi_ply(pretrained_ply_path_list)

    gaussians.training_setup(opt)
    gaussians.set_freeze("_opacity", True)
    gaussians.set_freeze("_rotation", True)
    gaussians.set_freeze("_scaling", True)
    gaussians.set_freeze("_xyz", True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_depth_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # Replace `opacity` with a maximum value
    # max_opacity = gaussians._opacity.max()
    # gaussians._opacity = torch.ones_like(gaussians._opacity) * max_opacity

    refined_save_dir = os.path.join(dataset.model_path, "refined_aligned")
    os.makedirs(refined_save_dir, exist_ok=True)
    iteration = 0
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt = viewpoint_cam.gt(release=False)

        gt_image = gt.image.to(device="cuda", non_blocking=True)
        gt_alpha = gt.alpha.to(device="cuda", non_blocking=True)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        # viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image: torch.Tensor = render_pkg["render"] # [3, H, W]
        rend_depth: torch.Tensor = render_pkg["render_depth"] # [1, H, W]

        mask = gt_alpha.expand_as(gt_image)
        masked_image = image * mask
        masked_gt = gt_image * mask

        # only RGB loss
        # Ll1 = l1_loss(image, gt_image)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        Ll1 = l1_loss(masked_image, masked_gt)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(masked_image, masked_gt))  
        # TODO: whethter to remove ssim loss

        depth_loss = torch.tensor(0.0, device="cuda")

        # if iteration > opt.depth_from_iter:
        #     if gt.depth_cam is not None:
        #         gt_depth = gt.depth_cam.to("cuda", non_blocking=True)
        #         mask = (gt_depth > 0.3) & (gt_depth < 7) & (rend_depth > 0.)
        #         # print(mask.shape)
        #         # print(depth.shape)
        #         # print(gt_depth.shape)
        #         # assert False
        #         depth_loss_sensor = l1_loss(rend_depth[mask], gt_depth[mask])
        #         depth_loss += opt.lambda_depth_sensor * depth_loss_sensor
        #         # print(depth.shape)
        #         # print(gt_depth.shape)
        #         # assert False

        #     if gt.depth_est is not None:
        #         dn_l1_weight = get_expon_lr_func(opt.dn_l1_weight_init, opt.dn_l1_weight_final, max_steps=opt.iterations)(iteration)
        #         pred_depth = gt.depth_est.to("cuda", non_blocking=True)
        #         mask = (rend_depth > 0.0) & (pred_depth > 0.0)
        #         # print(mask.shape)
        #         # assert False
        #         pred_depth_normalize = depth_normalize_(pred_depth[mask])
        #         rend_depth_normalize = depth_normalize_(rend_depth[mask])
        #         depth_loss_heuristic = l1_loss(pred_depth_normalize, rend_depth_normalize)
        #         depth_loss += 8 * dn_l1_weight * depth_loss_heuristic

        #         if iteration > opt.depth_from_iter + 1000:
        #             with torch.no_grad():
        #                 pred_depth_normal = depth_to_normal(viewpoint_cam, pred_depth).permute(2, 0, 1)

        #             # depth_normal_loss = l1_loss(pred_depth_normal, rend_depth_normal)
        #             # render_normal_loss = l1_loss(pred_depth_normal, render_normal)
        #             depth_normal_loss = (1 - (surf_normal * pred_depth_normal).sum(dim=0)).mean()
        #             depth_loss += dn_l1_weight * depth_normal_loss

        #     # isotropic regularization
        #     if opt.lambda_isotropic > 0:
        #         reg_loss = isotropic_loss(gaussians.get_scaling)
        #         loss += opt.lambda_isotropic * reg_loss

        # loss += depth_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_depth_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "depth": f"{ema_depth_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Densification forbidden for post refine 
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = opt.max_screen_size if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, max_screen_size=size_threshold)

            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

    shutil.rmtree(refined_save_dir, ignore_errors=True)
    os.makedirs(refined_save_dir)
    print("\n[ITER {}] Saving Gaussians".format(iteration))
    tmp_size = 0
    for gs_size, ply_name in zip(gs_size_list, pretrained_ply_name_list):
        mask3d = torch.zeros((len(gaussians._xyz), ), dtype=torch.bool, device="cuda")
        mask3d[tmp_size:tmp_size + gs_size] = True
        tmp_size += gs_size
        apply_mask3d(gaussians, mask3d, os.path.join(refined_save_dir, ply_name), return_clone_gs=False)
    gaussians.save_ply(os.path.join(refined_save_dir, "refined_{}.ply".format(iteration)))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--mask_dir", type=str, default = "/ssd1/liuziling/object_refine/gs-objects/data/3_17/masks_bin/masked")
    # parser.add_argument("--refined_object_dir", type=str, default = "generated_aligned")
    # parser.add_argument("--remained_background_path", type=str, default = "/ssd1/liuziling/object_refine/gs-objects/data/3_17/gs_seg/remained.ply")

    args = parser.parse_args(sys.argv[1:])

    # Initialize system state (RNG)
    safe_state(args.quiet)

    print("Optimizing " + args.model_path)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
    )

    # All done
    print("\nTraining complete.")
