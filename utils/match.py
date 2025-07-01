import sys
sys.path.append("submodules/mast3r")
# sys.path.append("submodules/mast3r/dust3r")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

from dust3r.inference import inference
from dust3r.utils.image import load_images_pil

# model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
# you can put the path to a local checkpoint in model_name if needed
__feature_extract_model = AsymmetricMASt3R.from_pretrained(
    "checkpoints/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    local_files_only=True,
).to("cuda")


def get_pairwise_mask3r_features(
    image1_pil: Image.Image,
    image2_pil: Image.Image,
    size=1024,
    device="cuda",
    n_points_per_cam=100,
):
    _images_pair = load_images_pil(
        [image1_pil.copy(), image2_pil.copy()],
        size=size,
        verbose=False,
    )
    _output = inference([tuple(_images_pair)], __feature_extract_model, device, batch_size=1, verbose=False)

    view1, pred1 = _output['view1'], _output['pred1']
    view2, pred2 = _output['view2'], _output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(
        desc1,
        desc2,
        subsample_or_initxy1=8,
        device=device,
        dist="dot",
        block_size=2**13,
    )

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    H1, W1 = view2['true_shape'][0]
    H0 = int(H0.item())
    W0 = int(W0.item())
    H1 = int(H1.item())
    W1 = int(W1.item())

    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    n_viz = n_points_per_cam
    num_matches = matches_im0.shape[0]
    n_viz = min(n_viz, num_matches)
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    # projecting indices to original coordinates
    viz_matches_im0_ = (viz_matches_im0 / np.array([[W0,H0]])*np.array([image1_pil.size])).astype(int)
    viz_matches_im1_ = (viz_matches_im1 / np.array([[W1,H1]])*np.array([image2_pil.size])).astype(int)

    return viz_matches_im0_, viz_matches_im1_  