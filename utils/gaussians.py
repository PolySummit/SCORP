import os
import sys
import numpy as np
import torch
from e3nn import o3
import einops
from einops import einsum

from utils.geometry import matrix_to_quaternion_tensor, quaternion_to_matrix_tensor

@torch.no_grad()
def gaussians_translate(gaussians, T: torch.Tensor):
    new_xyz = gaussians.get_xyz
    new_xyz = new_xyz + T[None]
    gaussians._xyz = new_xyz

@torch.no_grad()
def gaussians_scale(gaussians, scale:torch.Tensor, fix_center=False):
    # scale gaussians potsition
    if fix_center:
        new_xyz = gaussians.get_xyz
        mean_xyz = torch.mean(new_xyz, 0)
        new_xyz = new_xyz - mean_xyz
        new_xyz = new_xyz * scale[None]
        gaussians._xyz = new_xyz + mean_xyz
    else:
        gaussians._xyz = gaussians._xyz * scale[None]

    # scale gaussians scale
    new_scaling = torch.exp(gaussians._scaling) * scale[None]
    gaussians._scaling = torch.log(new_scaling)

def gaussians_rotate(gaussians,R:torch.Tensor, fix_center=False):
    rotate_xyz(gaussians, R, fix_center=fix_center)
    rotate_rot(gaussians, R)
    if gaussians.max_sh_degree == 0:
        return
    elif gaussians.max_sh_degree == 3:
        rotate_shs(gaussians, R)
    else:
        raise NotImplementedError(f"max_sh_degree={gaussians.max_sh_degree} is not supported")

@torch.no_grad()
def rotate_xyz(gaussians, R: torch.Tensor, fix_center=False):
    if fix_center:
        new_xyz = gaussians.get_xyz
        mean_xyz = torch.mean(new_xyz, 0)
        new_xyz = new_xyz - mean_xyz
        new_xyz = new_xyz @ R.T
        gaussians._xyz = new_xyz + mean_xyz
    else:
        gaussians._xyz = gaussians._xyz @ R.T


@torch.no_grad()
def rotate_rot(gaussians, R: torch.Tensor):
    #gaussians._xyz = torch.einsum("ij,nj->ni", R, gaussians._xyz)
    _rotation = quaternion_to_matrix_tensor(gaussians._rotation)
    _rotation = torch.einsum("ij,njk->nik", R, _rotation)
    gaussians._rotation = matrix_to_quaternion_tensor(_rotation)


@torch.no_grad()
def rotate_shs(gaussians, R: torch.Tensor):
    # reference: https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2147223570
    shs_feat = gaussians._features_rest
    ## rotate shs
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32) # switch axes: yzx -> xyz
    permuted_rotmat = np.linalg.inv(P) @ R.to("cpu").numpy() @ P
    rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotmat))

    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2]).to(shs_feat.device).float()
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2]).to(shs_feat.device).float()
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2]).to(shs_feat.device).float()

    # rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
            D_1,
            one_degree_shs,
            "... i j, ... j -> ... i",
        )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    two_degree_shs = einsum(
            D_2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
    two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 3:8] = two_degree_shs

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    three_degree_shs = einsum(
            D_3,
            three_degree_shs,
            "... i j, ... j -> ... i",
        )
    three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 8:15] = three_degree_shs

    gaussians._features_rest = shs_feat.float()


# def test():
#     ply_path = "./pretrained_2dgs/objects_office/2_18_declined/refined/cup.ply"
#     gs = GaussianModel(0)
#     gs.load_ply(ply_path)

#     print(gs._features_dc.shape)
#     print(gs._features_rest.shape)

def test_quaternion_to_matrix():
    q = torch.tensor([0.5, 0.5, 0.5, 0.5])
    R = quaternion_to_matrix_tensor(q)
    print(R)
    q_rec = matrix_to_quaternion_tensor(R)
    print(q_rec)

def test_matrix_to_quaternion():
    R = torch.eye(3,dtype=torch.float32)
    q = matrix_to_quaternion_tensor(R)
    print(q)

def test_scale_transform_2dgs():
    from gs2dgs.scene.gaussian_model import GaussianModel

    gs = GaussianModel(3)
    gs.load_ply("pretrained_2dgs_genfusion/04_27/Bench/easy/point_cloud/iteration_7000/point_cloud.ply")
    gaussians_scale(gs, torch.tensor(3.9972986496103893, device="cuda"))

    R = torch.tensor(
            [
                [0.3047, -0.9320, -0.1964],
                [-0.9465, -0.3194, 0.0469],
                [-0.1065, 0.1716, -0.9794],
            ],
            device="cuda",
        )

    # print(torch.det(R))

    gaussians_rotate(gs, R)
    gs.save_ply("./tmp_2dgs.ply")


if __name__=="__main__":
    test_scale_transform_2dgs()
