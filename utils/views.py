import json
import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
import sys
from PIL import Image
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Add parent directory to Python path
# sys.path.insert(0, os.path.dirname(current_dir))
from utils.geometry import quaternion_to_matrix_np
# from gs3dgs.scene.colmap_loader import read_extrinsics_text
from scipy.spatial.distance import cdist
import struct

img_extentions = ['.png', '.jpg', '.jpeg']


def get_xyxy_from_mask(mask: np.ndarray):
    """
    Obtain the bounding box coordinates (x1, y1, x2, y2) from a binary mask.
    :param mask: np.ndarray [H, W]
    :return: (x1, y1, x2, y2)
    """
    indices = np.where(mask > 0)

    x_indices = indices[1]
    y_indices = indices[0]

    if mask.sum() == 0:
        raise ValueError("No non-zero pixels found in the mask.")

    x1, x2 = x_indices.min(), x_indices.max()
    y1, y2 = y_indices.min(), y_indices.max()

    return (x1, y1, x2, y2)

def post_process_rgba_imgs(rgba_imgs:list[Image.Image]):
    crop_rgba_img_pil_list:list[Image.Image] = []
    # Crop the image with the minimum bounding box to remove excess background
    for rgba_img in rgba_imgs:
        rgba_img = np.array(rgba_img)
        alpha = rgba_img[:, :, 3]
        left, top, right, bottom = get_xyxy_from_mask(alpha)
        crop_rgba_img = rgba_img[top : bottom + 1, left : right + 1, :]
        crop_rgba_img_pil = Image.fromarray(crop_rgba_img)
        while crop_rgba_img_pil.size[0] * crop_rgba_img_pil.size[1] < 10000:
            crop_rgba_img_pil = crop_rgba_img_pil.resize(map(lambda x: int(x * 2), crop_rgba_img_pil.size))
        crop_rgba_img_pil_list.append(crop_rgba_img_pil)

    return crop_rgba_img_pil_list

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def parse_colmap_binary(path_to_model_file):
    """
    Read camera extrinsics from a COLMAP binary file, with output format consistent with parse_colmap_data
    
    Returns:
        camera_poses: Camera position array (N,3)
        camera_rot: Camera rotation quaternion array (N,4)
        image_names: List of image names (N,)
    """
    camera_poses = []
    camera_rot = []
    image_names = []
    
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            
            # Extract camera parameters
            qvec = np.array(binary_image_properties[1:5])  # rotation quaternion
            tvec = np.array(binary_image_properties[5:8])  # translation vector
            
            # Read image name
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            # Skip 2D point data (this information is not needed)
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            _ = read_next_bytes(fid, num_bytes=24*num_points2D, format_char_sequence="ddq"*num_points2D)
            
            # Store data
            camera_poses.append(tvec)
            camera_rot.append(qvec)
            image_names.append(image_name.rsplit(".", 1)[0])
    
    return np.array(camera_poses), np.array(camera_rot), image_names

def parse_colmap_data(colmap_path):
    """
        Parse Colmap camera pose data to extract positions of the cameras.
        Assume the file contains lines in the format:
        image_id, qw, qx, qy, qz, tx, ty, tz
    """
    t_W2C = []
    image_names = []
    q_W2C = []
    with open(colmap_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                        tuple(map(float, elems[1::3]))])
                t_W2C.append(tvec)
                image_names.append(image_name.rsplit(".", 1)[0])
                q_W2C.append(qvec)

    return np.array(t_W2C), np.array(q_W2C), image_names

def parse_transform_data(
    transform_path, isOpenGL=False
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract camera poses from a transform.json file.
    Args:
        transform_path (str): Path to the transform.json file.
        isOpenGL (bool): Whether the transformation is in OpenGL format.
    Returns:
        tuple: A tuple containing:
            - T_C2W (np.ndarray): Camera positions in world coordinates.
            - R_C2W (np.ndarray): Camera rotations in world coordinates.
            - image_names (list[str]): List of image names.
    """
    T_C2W = []
    image_names = []
    R_C2W = []
    with open(transform_path, "r") as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        for frame in frames:
            image_name = frame["file_path"].split("/")[-1]
            # Read rotation matrix and translation vector
            transform = np.array(frame["transform_matrix"])

            if isOpenGL:
                transform[:3, 1:3] *= -1

            # Extract rotation matrix and translation vector
            R = transform[:3, :3]
            T = transform[:3, 3]
            T_C2W.append(T)
            R_C2W.append(R)
            image_names.append(image_name)

    return np.array(T_C2W), np.array(R_C2W), image_names

def calculate_image_quality(img):
    """Improved image quality assessment (considering both sharpness and mask integrity)"""
    # Sharpness assessment
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Mask integrity assessment
    mask_completeness = evaluate_mask_ratio(img)

    # Comprehensive score (adjustable weights)
    return 0.3 * sharpness + 0.7 * mask_completeness


def evaluate_mask_shape_quality(alpha: np.ndarray):
    """
    Evaluate mask integrity using shape features, focusing on whether the target forms a complete closed region.
    """
    contours, _ = cv2.findContours((alpha == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    area = sum(cv2.contourArea(c) for c in contours)
    hull = cv2.convexHull(np.vstack(contours))
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return area / hull_area  # The closer to 1, the more complete and compact

def evaluate_mask_ratio(rgba_img: np.ndarray):
    """Evaluate mask integrity"""
    alpha = rgba_img[:, :, 3]
    valid_pixels = np.count_nonzero(alpha == 255)
    return valid_pixels / alpha.size


def viewpoint_diversity_score(t_W2C, R_W2C, current_idx, used_indices):
    """Calculate viewpoint diversity score"""
    if not used_indices:
        return 1.0

    t_C2W = -(t_W2C[:, :, None] * R_W2C).sum(axis=1)
    # t_C2W = -np.einsum("ni,nij->nj", t_W2C, R_W2C)

    # Position difference
    pos_distances = cdist([t_C2W[current_idx]], t_C2W[used_indices])
    pos_score = np.min(pos_distances)

    # Orientation difference (quaternion angle)
    current_rot = R_W2C[current_idx]
    dot_products = np.dot(R_W2C[used_indices], current_rot)
    angle_scores = 1 - np.abs(dot_products)  # The smaller the quaternion dot product, the greater the difference
    rot_score = np.min(angle_scores)

    return 0.5 * pos_score + 0.5 * rot_score

def viewpoint_diversity_score_paper(t_W2C, R_W2C, current_idx, used_indices):
    """Calculate viewpoint diversity score using relative normalization"""

    if not used_indices:
        return 1.0

    # World-to-camera pose --> Camera position in world frame
    t_C2W = -(t_W2C[:, :, None] * R_W2C).sum(axis=1)

    # ========== Position difference ==========
    all_pos_distances = cdist([t_C2W[current_idx]], t_C2W)[0]  # all distances
    used_pos_distances = all_pos_distances[used_indices]
    pos_score_raw = np.min(used_pos_distances)

    # min-max normalization
    pos_min, pos_max = np.min(all_pos_distances), np.max(all_pos_distances)
    pos_score = (pos_score_raw - pos_min) / (pos_max - pos_min + 1e-8)

    # ========== Direction difference ==========
    current_z = R_W2C[current_idx][:, 2]  # z-axis is the viewing direction
    all_z = R_W2C[:, :, 2]
    # normalize z
    all_z = all_z / np.linalg.norm(all_z, axis=1)[:, None]
    current_z = current_z / np.linalg.norm(current_z)
    
    dot_all = np.dot(all_z, current_z)  # dot product of all z-axis directions
    dot_used = dot_all[used_indices]
    angle_score_raw = np.min(1 - np.abs(dot_used))  # The more inconsistent, the higher the score

    # angle_min = np.min(1 - np.abs(dot_all))
    # angle_max = np.max(1 - np.abs(dot_all))
    # rot_score = (angle_score_raw - angle_min) / (angle_max - angle_min + 1e-8)

    # Comprehensive score
    return 0.5 * pos_score + 0.5 * angle_score_raw


def viewpoint_diversity_score_beta(t_W2C, R_W2C, current_idx, used_indices):
    """Calculate viewpoint diversity score"""
    """Rotation score only considers the z-axis, i.e., the camera viewing direction"""
    if not used_indices:
        return 1.0

    t_C2W = -(t_W2C[:, :, None] * R_W2C).sum(axis=1)
    # t_C2W = -np.einsum("ni,nij->nj", t_W2C, R_W2C)

    # Position difference
    pos_distances = cdist([t_C2W[current_idx]], t_C2W[used_indices])
    pos_score = np.min(pos_distances)

    # Orientation difference (quaternion angle)
    R_C2W = np.transpose(R_W2C, (0, 2, 1))
    current_z = R_W2C[current_idx][:, 2]
    used_z = R_C2W[used_indices][:, 2]
    dot_products = np.dot(used_z, current_z) # dot product of z-axis directions
    angle_scores = 1 - np.abs(dot_products)  # The greater the direction difference, the higher the score, and the more likely it is to be selected
    rot_score = np.min(angle_scores)
    # current_rot = R_W2C[current_idx]
    # dot_products = np.dot(R_W2C[used_indices], current_rot)
    # angle_scores = 1 - np.abs(dot_products)  # The smaller the quaternion dot product, the greater the difference
    # rot_score = np.min(angle_scores)

    return rot_score

def select_high_quality_and_diverse_images_gamma(
    rgba_dir,
    image_names,
    img_num,
    t_W2C,
    q_W2C=None,
    quality_weight=0.25,
    mask_weight=0.35,
    diversity_weight=0.4,
    R_W2C=None,
) -> tuple[list[int], list[np.ndarray]]:

    if R_W2C is None:
        if q_W2C is None:
            raise ValueError("Either q_W2C or R_W2C must be provided.")
        R_W2C = quaternion_to_matrix_np(q_W2C)

    # Read images and calculate various metrics
    rgba_imgs = []
    quality_scores = []
    mask_scores = []

    for img_name in image_names:
        img = cv2.imread(os.path.join(rgba_dir, f"{img_name}.png"), cv2.IMREAD_UNCHANGED)
        rgba_imgs.append(img)
        quality_scores.append(calculate_image_quality(img))
        mask_scores.append(np.sqrt(evaluate_mask_ratio(img)))

    # Normalize scores, requiring relative goodness
    quality_scores = (quality_scores - np.min(quality_scores)) / (np.max(quality_scores) - np.min(quality_scores))
    mask_scores = (mask_scores - np.min(mask_scores)) / (np.max(mask_scores) - np.min(mask_scores))

    # Initialize selection
    selected_indices = []
    remaining_indices = set(range(len(rgba_imgs)))

    while len(selected_indices) < img_num and remaining_indices:
        best_score = -np.inf
        best_idx = -1

        # Iterate through remaining candidate images
        for idx in remaining_indices:
            # Base score
            base_score = (
                quality_weight * quality_scores[idx] + mask_weight * mask_scores[idx]
            )

            # Diversity score
            if selected_indices:
                div_score = viewpoint_diversity_score(
                    t_W2C,
                    R_W2C,
                    idx,
                    selected_indices,
                )
            else:
                div_score = 1.0

            total_score = base_score + diversity_weight * div_score

            if total_score > best_score:
                best_score = total_score
                best_idx = idx

        # Update selection
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

        # Remove nearby viewpoints (dynamic threshold)
        # if len(selected_indices) < img_num:
        #     current_pos = t_W2C[best_idx]
        #     distances = cdist([current_pos], t_W2C).flatten()
        #     median_dist = np.median(distances)
        #     nearby_indices = set(np.where(distances < median_dist * 0.4)[0])
        #     remaining_indices -= nearby_indices

    return (
        [idx for idx in selected_indices],
        [rgba_imgs[idx] for idx in selected_indices],
    )

def select_high_quality_and_diverse_images_beta(
    rgba_dir,
    image_names,
    img_num,
    t_W2C,
    q_W2C=None,
    quality_weight=0.25,
    mask_weight=0.5,
    diversity_weight=0.25,
    R_W2C=None,
    mask_threshold=0.3,
) -> list[np.ndarray]:

    if R_W2C is None:
        if q_W2C is None:
            raise ValueError("Either q_W2C or R_W2C must be provided.")
        R_W2C = quaternion_to_matrix_np(q_W2C)

    # Read images and calculate various metrics
    rgba_imgs = []
    quality_scores = []
    mask_scores = []
    mask_completeness_scores = []
    for img_name in image_names:
        # img = cv2.imread(os.path.join(rgba_dir, f"{img_name}.png"), cv2.IMREAD_UNCHANGED)
        img = np.array(Image.open(os.path.join(rgba_dir, f"{img_name}.png")))
        rgba_imgs.append(img)
        mask_completeness_score = evaluate_mask_shape_quality(img[:, :, 3])
        quality_scores.append(calculate_image_quality(img))
        mask_scores.append(evaluate_mask_ratio(img))
        mask_completeness_scores.append(mask_completeness_score)
        # mask_completeness_scores.append(0)

    # Normalize scores, requiring relative goodness
    quality_scores = (quality_scores - np.min(quality_scores)) / (np.max(quality_scores) - np.min(quality_scores))
    mask_scores = (mask_scores - np.min(mask_scores)) / (np.max(mask_scores) - np.min(mask_scores))

    # Calculate threshold (by percentile)
    threshold_value = np.quantile(mask_scores, mask_threshold)

    # Set mask_scores below the threshold to 0
    mask_scores[mask_scores < threshold_value] = 0

    # Discard images where the mask ratio is too small
    removed_indices = np.where(mask_scores == 0)[0]

    # Initialize selection
    selected_indices = []
    remaining_indices = set(range(len(rgba_imgs)))

    # For images with too small a mask, the trellis generation effect is not good
    for idx in removed_indices:
        if idx in remaining_indices:
            remaining_indices.remove(idx)

    while len(selected_indices) < img_num and remaining_indices:
        best_score = -np.inf
        best_idx = -1

        # Iterate through remaining candidate images
        for idx in remaining_indices:
            # Base score
            base_score = (
                quality_weight * quality_scores[idx]
                + mask_weight * mask_scores[idx]
                + 0.4 * mask_completeness_scores[idx]
            )

            # Diversity score
            if selected_indices:
                # div_score = viewpoint_diversity_score_beta(
                #     t_W2C,
                #     R_W2C,
                #     idx,
                #     selected_indices,
                # )
                div_score = viewpoint_diversity_score_paper(
                    t_W2C,
                    R_W2C,
                    idx,
                    selected_indices,
                )

            else:
                div_score = 1.0

            total_score = base_score + diversity_weight * div_score

            if total_score > best_score:
                best_score = total_score
                best_idx = idx

        # Update selection
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

        # Remove nearby viewpoints (dynamic threshold)
        # if len(selected_indices) < img_num:
        #     current_pos = t_W2C[best_idx]
        #     distances = cdist([current_pos], t_W2C).flatten()
        #     median_dist = np.median(distances)
        #     nearby_indices = set(np.where(distances < median_dist * 0.4)[0])
        #     remaining_indices -= nearby_indices

    return [rgba_imgs[idx] for idx in selected_indices]

def select_high_quality_and_diverse_images(
    rgba_dir,
    colmap_path,
    img_num,
    quality_weight=0.25,
    mask_weight=0.25,
    diversity_weight=0.25,
) -> list[np.ndarray]:
    """
    Select high-quality and diverse images based on camera pose data and image quality.
    
    Args:
        colmap_path (str): Path to the Colmap camera pose data file.
        img_num (int): Number of images to select.
    
    Returns:
        list: Selected RGBA images.
    """
    if colmap_path.endswith(".bin"):
        t_W2C, q_W2C, image_names = parse_colmap_binary(colmap_path)
    elif colmap_path.endswith(".txt"):
        t_W2C, q_W2C, image_names = parse_colmap_data(colmap_path)

    return select_high_quality_and_diverse_images_beta(
        rgba_dir,
        image_names,
        img_num,
        t_W2C,
        q_W2C,
        quality_weight=quality_weight,
        mask_weight=mask_weight,
        diversity_weight=diversity_weight,
    )


def merge_to_rgba(img_dir, mask_dir, output_dir):
    """
    Merge RGB images and label masks to RGBA images.
    """
    mask_imgs = os.listdir(mask_dir)
    mask_imgs = [img for img in mask_imgs if os.path.splitext(img)[1] in img_extentions]
    mask_imgs.sort()
    rgb_imgs = os.listdir(img_dir)
    rgb_imgs = [img for img in rgb_imgs if os.path.splitext(img)[1] in img_extentions]
    rgb_imgs.sort()
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # else:
    #     files = os.listdir(output_dir)
    #     if len(files) == len(mask_imgs):
    #         print(f"RGBA images already exist in {output_dir}")
    #         return
    
    for i, mask_img in tqdm(enumerate(mask_imgs), desc="Merging to RGBA", total=len(mask_imgs)):
        mask = cv2.imread(os.path.join(mask_dir, mask_img), cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.imread(os.path.join(img_dir, rgb_imgs[i]))
        alpha = np.zeros_like(mask)
        alpha[mask > 0] = 255
        rgba_img = cv2.merge([rgb_img, alpha])
        cv2.imwrite(os.path.join(output_dir, rgb_imgs[i]), rgba_img)

def main():
    parser = argparse.ArgumentParser(description="select viewpoints")
    parser.add_argument("--rgba_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, help="save selected images", default="")
    parser.add_argument("--colmap_path", type=str, default="")
    parser.add_argument("--img_num", type=int, default=4)
    args = parser.parse_args()
    # os.makedirs(args.rgba_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.rgba_dir):
        raise ValueError(f"RGBA directory {args.rgba_dir} does not exist")
    # # merge to rgba
    # rgba_files = os.listdir(args.rgba_dir)
    # if len(rgba_files) == 0:
    #     print("Merging RGB and mask images to RGBA...")
    #     # merge rgb and mask
    #     # merge_to_rgba(args.img_dir, args.mask_dir, args.rgba_dir)
    # else:
    #     print(f"RGBA images already exist in {args.rgba_dir}")
    # select viewpoints
    # read_extrinsics_text(args.colmap_path)
    selected_rgba = select_high_quality_and_diverse_images(args.rgba_dir, args.colmap_path, args.img_num)
    # Crop out the center of the mask
    crop_rgba_imgs = post_process_rgba_imgs(selected_rgba)

    # save selected images to output_dir
    for i, rgba_img in enumerate(crop_rgba_imgs):
        # Image.fromarray(rgba_img).save(os.path.join(args.output_dir, f"img_{i}.png"))
        cv2.imwrite(os.path.join(args.output_dir, f"img_{i}.png"), rgba_img)
    print(f"Saved {len(crop_rgba_imgs)} images to {args.output_dir}")


if __name__ == '__main__':
    main()
