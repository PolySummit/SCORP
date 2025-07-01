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
import sys
from PIL import Image
from typing import Callable, NamedTuple

import yaml
from .colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from ..utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from ..utils.sh_utils import SH2RGB
from .gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: np.ndarray
    FovX: np.ndarray
    depth_cam_path: str
    depth_est_path: str
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list[CameraInfo]
    test_cameras: list[CameraInfo]
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(
    cam_extrinsics,
    cam_intrinsics,
    images_folder,
    depth_cam_folder=None,
    depth_est_folder=None,
):
    cam_infos: list[CameraInfo] = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        if not os.path.exists(image_path):
            image_path = image_path.rsplit(".", 1)[0] + ".png"

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")

        depth_cam_path = None
        if depth_cam_folder is not None:
            depth_cam_path = os.path.join(depth_cam_folder, image_name)

        depth_est_path = None
        if depth_est_folder is not None:
            depth_est_path = os.path.join(depth_est_folder, image_name)

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            depth_cam_path=depth_cam_path,
            depth_est_path=depth_est_path,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=None)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, split_yml_name=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        depth_cam_folder=os.path.join(path, "depths_cam") if os.path.exists(os.path.join(path, "depths")) else None,
        depth_est_folder=os.path.join(path, "depths_est") if os.path.exists(os.path.join(path, "depths_est")) else None
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        split_file = os.path.join(path, split_yml_name)
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found at {split_file}")
        print("Reading split file")
        with open(split_file, "r") as f:
            split = yaml.safe_load(f)
        train_list = split["train"]
        test_list = split["test"]
        train_cam_infos = [c for c in cam_infos if c.image_name in train_list]
        test_cam_infos = [c for c in cam_infos if c.image_name in test_list]
        # else:
        #     print("Split file not found, using LLFF holdout")
        #     raise NotImplementedError("LLFF holdout not implemented for this work")
        #     # NOTE: This is a hack to make sure that the same cameras are used for training and testing
        #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    print(f"Train cameras: {len(train_cam_infos)}, Test cameras: {len(test_cam_infos)}")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/points3D.ply")
    bin_path = os.path.join(path, "sparse/points3D.bin")
    txt_path = os.path.join(path, "sparse/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(
    path,
    images_dir,
    transformsfile,
    white_background,
    depth_cam_folder=None,
    depth_est_folder=None,
    isOpenGL=False,
    extension=".jpg",
)->list[CameraInfo]:
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(images_dir, frame["file_path"].rsplit("/",1)[1])
            if os.path.exists(cam_name+extension):
                cam_name += extension
            else:
                cam_name += ".png"

            if not os.path.exists(cam_name):
                raise FileNotFoundError(f"Image file not found at {cam_name}")

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            if isOpenGL:
                c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_cam_path = None
            if depth_cam_folder is not None:
                depth_cam_path = os.path.join(depth_cam_folder, image_name)

            depth_est_path = None
            if depth_est_folder is not None:
                depth_est_path = os.path.join(depth_est_folder, image_name)

            cam_info = CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                depth_cam_path=depth_cam_path,
                depth_est_path=depth_est_path,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
            )

            cam_infos.append(cam_info)

    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, isOpenGL=True, extension=extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, isOpenGL=True, extension=extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readToyDeskSceneInfo(path, images, white_background, eval, extension=".jpg", split_yml_name=None):
    print("Reading Transforms")
    print(path)

    images_dir = "images" if images == None else images

    cam_infos = readCamerasFromTransforms(
        path,
        os.path.join(path, images_dir),
        "transforms_full.json",
        white_background,
        depth_cam_folder=(
            os.path.join(path, "depths")
            if os.path.exists(os.path.join(path, "depths"))
            else None
        ),
        depth_est_folder=(
            os.path.join(path, "depths_est")
            if os.path.exists(os.path.join(path, "depths_est"))
            else None
        ),
        extension=extension,
    )

    nerf_normalization = getNerfppNorm(cam_infos)

    # Get 3D bbox from cam_infos

    cam_centers = np.stack([cam_info.T for cam_info in cam_infos])
    cam_centers_min = cam_centers.min(axis=0)
    cam_centers_max = cam_centers.max(axis=0)

    cam_centers_radius = np.linalg.norm(cam_centers_max - cam_centers_min) / 2.0
    cam_centers_center = (cam_centers_max + cam_centers_min) / 2.0
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 360_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of 3D bbox
        # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        xyz = (np.random.random((num_pts, 3)) - 0.5) * 4 * cam_centers_radius + cam_centers_center
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    pcd = fetchPly(ply_path)


    if eval:
        split_file = os.path.join(path, split_yml_name)
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found at {split_file}")
        print("Reading split file")
        with open(split_file, "r") as f:
            split = yaml.safe_load(f)
        train_list = split["train"]
        test_list = split["test"]
        train_cam_infos = [c for c in cam_infos if c.image_name in train_list]
        test_cam_infos = [c for c in cam_infos if c.image_name in test_list]
        # else:
        #     print("Split file not found, using LLFF holdout")
        #     # NOTE: This is a hack to make sure that the same cameras are used for training and testing
        #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks: dict[str, Callable[..., SceneInfo]] = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "ToyDesk": readToyDeskSceneInfo,
}
