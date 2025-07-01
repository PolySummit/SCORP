from typing import Literal
import numpy as np
import torch

from utils.geometry import quaternion_to_matrix_tensor, matrix_to_quaternion_tensor


def kabsch_algorithm_np(
    P: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes the optimal translation and rotation matrices that minimize the 
    RMS deviation between two sets of points P and Q using Kabsch's algorithm.
    More here: https://en.wikipedia.org/wiki/Kabsch_algorithm
    Inspiration: https://github.com/charnley/rmsd
    
    inputs: P  N x 3 numpy matrix representing the coordinates of the points in P
            Q  N x 3 numpy matrix representing the coordinates of the points in Q
            
    return: R, t  the optimal rotation matrix and translation vector
    """
    if (P.size == 0 or Q.size == 0):
        raise ValueError("Empty matrices sent to kabsch")
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    P_centered = P - centroid_P                       # Center both matrices on centroid
    Q_centered = Q - centroid_Q
    H = P_centered.T.dot(Q_centered)                  # covariance matrix
    U, S, VT = np.linalg.svd(H)                        # SVD
    R = U.dot(VT).T                                    # calculate optimal rotation

    if np.linalg.det(R) < 0:                          # correct rotation matrix for             
        VT[2,:] *= -1                                  #  right-hand coordinate system
        R = U.dot(VT).T                          
    t = centroid_Q - R.dot(centroid_P)                # translation vector

    return R, t, 1.0


def umeyama_algorithm_np(
    source_points,
    target_points,
) -> tuple[np.ndarray, np.ndarray, float]:
    '''
    source_points: np.ndarray, shape=(n, 3)
    target_points: np.ndarray, shape=(n, 3)
    return: R, t, s
    '''
    if len(source_points) != len(target_points):
        raise ValueError("Source and target points must have the same length")

    # print("num source_points:", len(source_points))

    # Compute centroids
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    # Center the points
    centered_source = source_points - source_centroid
    centered_target = target_points - target_centroid

    # Compute covariance matrix
    cov = centered_source.T @ centered_target

    # Perform SVD
    U, S, Vt = np.linalg.svd(cov)
    V = Vt.T

    # Create correction matrix D to handle right-handedness
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1

    # Compute rotation
    R = V @ D @ U.T

    # Compute scale
    s = np.sum(S * np.diag(D)) / np.sum(centered_source ** 2)

    # Compute translation
    t = target_centroid - s * (R @ source_centroid)

    return R, t, s


def umeyama_algorithm_generalized_np(
    source_points: np.ndarray, target_points: np.ndarray
):
    '''
    source_points: np.ndarray, shape=(n, 3)
    target_points: np.ndarray, shape=(n, 3)
    return: R, t, S
    '''
    # Compute centroids
    source_centroid: np.ndarray = np.mean(source_points, axis=0)  # Mean along columns, assuming (3, N) shape
    target_centroid: np.ndarray = np.mean(target_points, axis=0)

    # Center the points by subtracting centroids
    q_B_center = source_points - source_centroid  # Broadcasting to subtract from each column
    q_A_center = target_points - target_centroid

    # Compute H and P matrices
    H = q_A_center.T @ q_B_center  # (3, N) @ (N, 3) = (3, 3)
    P = q_B_center.T @ q_B_center

    # Singular Value Decomposition (SVD) of H and P
    # UH, SH, VH = np.linalg.svd(H)  # VH is V transpose
    # UP, SP, VP = np.linalg.svd(P)

    # Reconstruct H_d and P_d from SVD components
    # H_d = UH @ np.diag(SH) @ VH
    # P_d = UP @ np.diag(SP) @ VP

    # Compute K_d as H_d times inverse of P_d
    K_d = H @ np.linalg.inv(P)

    # SVD of K_d
    U, S, Vh = np.linalg.svd(K_d)  # Vh is V transpose

    # Create correction matrix D to handle right-handedness
    D = np.eye(3)
    if np.linalg.det(U @ Vh) < 0:
        D[-1, -1] = -1

    # Compute rotation matrix
    rotation = U @ D @ Vh

    # Define diagonal matrices for scale computation
    D1 = np.diag([1, 0, 0])
    D2 = np.diag([0, 1, 0])
    D3 = np.diag([0, 0, 1])

    # Compute scale factors s1, s2, s3
    s1 = np.trace(q_A_center @ rotation @ D1 @ q_B_center.T) / np.trace(q_B_center @ D1 @ q_B_center.T)
    s2 = np.trace(q_A_center @ rotation @ D2 @ q_B_center.T) / np.trace(q_B_center @ D2 @ q_B_center.T)
    s3 = np.trace(q_A_center @ rotation @ D3 @ q_B_center.T) / np.trace(q_B_center @ D3 @ q_B_center.T)

    # Construct scale as a diagonal matrix
    scale = np.array([s1, s2, s3])

    # Compute translation vector
    translation = target_centroid - rotation @ (scale * source_centroid)

    # Return the results as a tuple
    return rotation, translation, scale

def polar_12dof_np(source_points: np.ndarray, target_points: np.ndarray): #####################
    '''
    source_points: np.ndarray, shape=(n, 3)
    target_points: np.ndarray, shape=(n, 3)
    return: R, t, S
    '''
    # Compute centroids
    src_centroid: np.ndarray = source_points.mean(axis=0)
    tgt_centroid: np.ndarray = target_points.mean(axis=0)

    # Center the points by subtracting centroids
    Bc = source_points - src_centroid   # (N,3)
    Ac = target_points - tgt_centroid   # (N,3)

    H = Ac.T @ Bc @ np.linalg.pinv(Bc.T @ Bc)

    M = H.T @ H
    eigval_M, eigvec_M = np.linalg.eigh(M)
    eigval_M = np.clip(eigval_M, 0.0, None) # For numerical stability
    sqrt_M = eigvec_M @ np.diag(np.sqrt(eigval_M)) @ eigvec_M.T

    # R_raw = H · U^{-1}
    R = H @ np.linalg.inv(sqrt_M)

    U_r, _, V_r = np.linalg.svd(R)
    R = U_r @ V_r

    if np.linalg.det(R) < 0:
        U_r[:, -1] *= -1
        R = U_r @ V_r

    eigval_U, eigvec_U = np.linalg.eigh(sqrt_M)
    eigval_U = np.clip(eigval_U, 1e-9, None)       # For positive definiteness
    R_prime = eigvec_U.T
    if np.linalg.det(R_prime) < 0:                 # For right-handedness
        eigvec_U[:, -1] *= -1
        R_prime = eigvec_U.T

    S = np.diag(eigval_U)

    t = tgt_centroid - R @ R_prime.T @ S @ R_prime @ src_centroid

    return R, t, S, R_prime

def adam_algorithm_3d2d_9dof(
    source_points_3d_world: np.ndarray,
    target_points_2d_image: np.ndarray,
    extrinsic_matrix: np.ndarray, # w2c
    intrinsic_matrix: np.ndarray,
    iterations: int = 1000,
    verbose_interval: int = 100,
    lr: float = 6e-3,
    lambda_reg_rot: float = 1e-5,
    scale_max: float = 1.5,
    scale_min: float = 0.75,
    init_rotation: np.ndarray = None,
    init_translation: np.ndarray = None,
    init_scale: float = 1.0,
    device="cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    source_points_3d_world_tensor = torch.tensor(source_points_3d_world, dtype=torch.float32, device=device)
    target_points_2d_image_tensor = torch.tensor(target_points_2d_image, dtype=torch.float32, device=device)
    extrinsic_matrix_tensor = torch.tensor(extrinsic_matrix, dtype=torch.float32, device=device)
    intrinsic_matrix_tensor = torch.tensor(intrinsic_matrix, dtype=torch.float32, device=device)

    if isinstance(init_scale, float):
        init_scale = np.array(3 * [init_scale])
    elif isinstance(init_scale, (list, tuple)):
        init_scale = np.array(init_scale)

    if not isinstance(init_scale, np.ndarray) or init_scale.shape != (3,):
        raise ValueError("`init_scale` must be a float, list, or tuple of length 3.")

    if init_scale.min() < scale_min or init_scale.max() > scale_max:
        init_scale = np.array(3 * [scale_min + (scale_max - scale_min) / 2])

    if init_rotation is not None:
        with torch.no_grad():
            init_rotation_q = matrix_to_quaternion_tensor(torch.tensor(init_rotation, dtype=torch.float32, device=device))
        param_quaternion = torch.nn.Parameter(init_rotation_q)
    else:
        param_quaternion = torch.nn.Parameter(torch.randn(3, dtype=torch.float32, device=device))

    if init_translation is not None:
        param_translation = torch.nn.Parameter(torch.tensor(init_translation, dtype=torch.float32, device=device))
    else:
        param_translation = torch.nn.Parameter(torch.randn(4, dtype=torch.float32, device=device))

    with torch.no_grad():
        init_scale_param = torch.logit(
            (torch.tensor(init_scale, dtype=torch.float32, device=device) - scale_min)
            / (scale_max - scale_min)
        )
    param_quaternion_orthogonal = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device))

    param_scale = torch.nn.Parameter(init_scale_param)

    optimizer = torch.optim.Adam([
        {"params": param_translation, "lr": lr},
        {"params": param_quaternion, "lr": lr},
        {"params": param_scale, "lr": lr},
    ])

    for iteration in range(iterations):
        current_scale = scale_min + (scale_max - scale_min) * torch.sigmoid(param_scale)
        current_rot = quaternion_to_matrix_tensor(param_quaternion)
        current_rot_orthogonal = quaternion_to_matrix_tensor(param_quaternion_orthogonal)
        current_points_3d_world = (
            (current_scale * (source_points_3d_world_tensor @ current_rot_orthogonal.T))
            @ current_rot_orthogonal
        ) @ current_rot.T + param_translation
        current_points_3d_camera = current_points_3d_world @ extrinsic_matrix_tensor[:3, :3].T + extrinsic_matrix_tensor[:3, 3]
        current_camera_depth = current_points_3d_camera[:, -1]
        current_points_2d_image = ((current_points_3d_camera @ intrinsic_matrix_tensor.T) / current_camera_depth[:, None])[:,:2]

        loss_opt = torch.mean((current_points_2d_image - target_points_2d_image_tensor) ** 2)

        reg_rot = (torch.arccos(torch.clamp((torch.trace(current_rot) - 1) / 2, -1, 1)) ** 2).mean()

        loss = loss_opt + lambda_reg_rot * reg_rot

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose_interval > 0 and (iteration + 1) % verbose_interval == 0:
            with torch.no_grad():
                print(f"Iteration {iteration + 1:4d} | Loss: {loss.item():.6f}")
                print(f"Current Scale: {current_scale.detach().cpu()}")

    # Return the results as a tuple
    with torch.no_grad():
        rotation = quaternion_to_matrix_tensor(param_quaternion).cpu().numpy()
        translation = param_translation.cpu().numpy()
        scale = scale_min + (scale_max - scale_min) * torch.sigmoid(param_scale)
    return rotation, translation, scale

def adam_algorithm_3d2d_7dof(
    source_points_3d_world: np.ndarray,
    target_points_2d_image: np.ndarray,
    extrinsic_matrix: np.ndarray, # w2c
    intrinsic_matrix: np.ndarray,
    iterations: int = 1000,
    verbose_interval: int = 100,
    lr: float = 6e-3,
    lambda_reg_rot: float = 1e-5,
    scale_max: float = 1.5,
    scale_min: float = 0.75,
    init_rotation: np.ndarray = None,
    init_translation: np.ndarray = None,
    init_scale: float = 1.0,
    device="cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    source_points_3d_world_tensor = torch.tensor(source_points_3d_world, dtype=torch.float32, device=device)
    target_points_2d_image_tensor = torch.tensor(target_points_2d_image, dtype=torch.float32, device=device)
    extrinsic_matrix_tensor = torch.tensor(extrinsic_matrix, dtype=torch.float32, device=device)
    intrinsic_matrix_tensor = torch.tensor(intrinsic_matrix, dtype=torch.float32, device=device)

    if init_rotation is not None:
        with torch.no_grad():
            init_rotation_q = matrix_to_quaternion_tensor(torch.tensor(init_rotation, dtype=torch.float32, device=device))
        param_quaternion = torch.nn.Parameter(init_rotation_q)
    else:
        param_quaternion = torch.nn.Parameter(torch.randn(3, dtype=torch.float32, device=device))

    if init_translation is not None:
        param_translation = torch.nn.Parameter(torch.tensor(init_translation, dtype=torch.float32, device=device))
    else:
        param_translation = torch.nn.Parameter(torch.randn(4, dtype=torch.float32, device=device))

    init_scale_param = torch.logit(
        (torch.tensor(init_scale, dtype=torch.float32, device=device) - scale_min)
        / (scale_max - scale_min)
    )
    param_scale = torch.nn.Parameter(init_scale_param)

    optimizer = torch.optim.Adam([
        {"params": param_translation, "lr": lr},
        {"params": param_quaternion, "lr": lr},
        {"params": param_scale, "lr": lr},
    ])

    for iteration in range(iterations):
        current_scale = scale_min + (scale_max - scale_min) * torch.sigmoid(param_scale)
        current_rot = quaternion_to_matrix_tensor(param_quaternion)
        current_points_3d_world = (current_scale * source_points_3d_world_tensor) @ current_rot.T + param_translation
        current_points_3d_camera = current_points_3d_world @ extrinsic_matrix_tensor[:3, :3].T + extrinsic_matrix_tensor[:3, 3]
        current_camera_depth = current_points_3d_camera[:, -1]
        current_points_2d_image = ((current_points_3d_camera @ intrinsic_matrix_tensor.T) / current_camera_depth[:, None])[:,:2]

        loss_opt = torch.mean((current_points_2d_image - target_points_2d_image_tensor) ** 2)

        reg_rot = (torch.arccos(torch.clamp((torch.trace(current_rot) - 1) / 2, -1, 1)) ** 2).mean()

        loss = loss_opt + lambda_reg_rot * reg_rot

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose_interval > 0 and (iteration + 1) % verbose_interval == 0:
            with torch.no_grad():
                print(f"Iteration {iteration + 1:4d} | Loss: {loss.item():.6f}")
                print(f"Current Scale: {current_scale.detach().cpu().item()}")

    # Return the results as a tuple
    with torch.no_grad():
        rotation = quaternion_to_matrix_tensor(param_quaternion).cpu().numpy()
        translation = param_translation.cpu().numpy()
        scale = scale_min + (scale_max - scale_min) * torch.sigmoid(param_scale).item()
    return rotation, translation, scale

def adam_algorithm_3d3d_9dof(
    source_points: np.ndarray,
    target_points: np.ndarray,
    iterations: int = 1000,
    verbose_interval: int = 100,
    lr: float = 1e-3,
    lambda_reg_scale: float = 2e-5,
    lambda_reg_rot: float = 1e-4,
    scale_max: float = 1.5,
    scale_min: float = 0.75,
    init_scale: float = 1.0,
    device="cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_points_tensor = torch.tensor(source_points, dtype=torch.float32, device=device)
    target_points_tensor = torch.tensor(target_points, dtype=torch.float32, device=device)

    if isinstance(init_scale, float):
        init_scale = np.array(3 * [init_scale])
    elif isinstance(init_scale, (list, tuple)):
        init_scale = np.array(init_scale)

    if not isinstance(init_scale, np.ndarray) or init_scale.shape != (3,):
        raise ValueError("`init_scale` must be a float, list, or tuple of length 3.")

    if init_scale.min() < scale_min or init_scale.max() > scale_max:
        init_scale = np.array(3 * [scale_min + (scale_max - scale_min) / 2])

    param_translation = torch.nn.Parameter(torch.tensor([0.01, 0.01, 0.01], dtype=torch.float32, device=device))
    param_quaternion = torch.nn.Parameter(torch.tensor([0.9, 0.01, 0.01, 0.01], dtype=torch.float32, device=device))
    param_quaternion_orthogonal = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device))

    with torch.no_grad():
        init_scale_param = torch.logit(
            (torch.tensor(init_scale, dtype=torch.float32, device=device) - scale_min)
            / (scale_max - scale_min)
        )
    param_scale = torch.nn.Parameter(init_scale_param)

    optimizer = torch.optim.Adam([
        {"params": param_translation, "lr": lr},
        {"params": param_quaternion, "lr": lr},
        {"params": param_quaternion_orthogonal, "lr": lr},
        {"params": param_scale, "lr": lr},
    ])

    for iteration in range(iterations):
        current_scale = scale_min + (scale_max - scale_min) * torch.sigmoid(param_scale)
        # print(current_scale)
        current_rot = quaternion_to_matrix_tensor(param_quaternion)
        current_rot_orthogonal = quaternion_to_matrix_tensor(param_quaternion_orthogonal)
        predicted_points = (
            (
                current_rot
                @ current_rot_orthogonal.T
                @ (current_scale[:, None] * (current_rot_orthogonal @ source_points_tensor.T))
            ).T
        ) + param_translation
        loss_opt = torch.mean((predicted_points - target_points_tensor) ** 2)

        mean_scale = torch.mean(current_scale)
        reg_scale_mean = torch.mean((current_scale - mean_scale) ** 2)

        reg_scale = torch.mean((param_scale - 1) ** 2)

        reg_rot = (torch.arccos(torch.clamp((torch.trace(current_rot) - 1) / 2, -1, 1)) ** 2).mean()

        loss = loss_opt + lambda_reg_scale * (reg_scale + reg_scale_mean) + lambda_reg_rot * reg_rot

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose_interval > 0 and (iteration + 1) % verbose_interval == 0:
            with torch.no_grad():
                print(f"Iteration {iteration + 1:4d} | Loss: {loss.item():.6f}")
                print(f"Current Scale: {current_scale.detach().cpu().numpy().tolist()}")

    # Return the results as a tuple
    with torch.no_grad():
        rotation = quaternion_to_matrix_tensor(param_quaternion).cpu().numpy()
        rotation_orthogonal = quaternion_to_matrix_tensor(param_quaternion_orthogonal).cpu().numpy()
        translation = param_translation.cpu().numpy()
        scale = scale_min + (scale_max - scale_min) * torch.sigmoid(param_scale).cpu().numpy()
    return rotation, translation, scale, rotation_orthogonal


def compute_residuals(
    source_points: np.ndarray, 
    target_points: np.ndarray, 
    R: np.ndarray, 
    t: np.ndarray, 
    s: np.ndarray = 1.0
) -> np.ndarray:
    """
    Computes the residuals between transformed source points and target points.

    Params:
        source_points: (N, 3) numpy array of source points
        target_points: (N, 3) numpy array of target points
        R: (3, 3) numpy array representing the rotation matrix
        t: (3,) numpy array representing the translation vector
        s: float representing the scale factor (default is 1.0)

    Returns:
        residuals: (N,) numpy array of residuals, where each element is the Euclidean distance
                   between the transformed source point and the corresponding target point.
    """
    transformed_source = (R @ (s * source_points).T).T + t
    diff = transformed_source - target_points
    residuals = np.linalg.norm(diff, axis=1)
    return residuals


def pc_align_ransac(
    source_points: np.ndarray, 
    target_points: np.ndarray,
    threshold: float = 0.5,
    max_iterations: int = 2000,
    min_inlier_ratio: float = -1.0,
    method: Literal["umeyama", "kabsch", "umeyama_gen"] = "umeyama"
):
    """
    RANSAC algorithm to find the best fitting transformation (rotation, translation, scale)
    between two sets of points using Umeyama or Kabsch algorithm.
    
    Params:
        source_points: (N, 3) numpy array of source points
        target_points: (N, 3) numpy array of target points
        threshold: float, distance threshold to consider a point as an inlier
        max_iterations: int, maximum number of iterations for RANSAC
        min_inlier_ratio: float, minimum ratio of inliers to consider a model valid
        
    Returns:
        best_R: (3, 3) numpy array representing the best rotation matrix
        best_t: (3,) numpy array representing the best translation vector
        best_s: float, representing the best scale factor
    """
    if len(source_points) != len(target_points):
        raise ValueError("Source and target points must have the same length")
    if len(source_points) < 3:
        raise ValueError("At least 3 points are required to solve Umeyama.")

    if method == "umeyama":
        solve_method = umeyama_algorithm_np
    elif method == "kabsch":
        solve_method = kabsch_algorithm_np
    elif method == "umeyama_gen":
        solve_method = umeyama_algorithm_generalized_np


    N = len(source_points)
    best_inliers = []
    # best_model = None  # (R, t, s)
    max_inlier_count = 0

    for i in range(max_iterations):
        # 1. Randomly sample 3 non-collinear point pairs (if the data distribution is unknown, just sample 3 randomly for now)
        #   Logic to check for collinearity can be added as needed
        sample_indices = np.random.choice(N, 3, replace=False)
        sp_sample = source_points[sample_indices]
        tp_sample = target_points[sample_indices]

        # 2. Use the sampled 3 pairs of points to solve with the selected method
        try:
            R_est, t_est, s_est = solve_method(sp_sample, tp_sample)
        except np.linalg.LinAlgError:
            # SVD failed or other numerical exception, skip this iteration
            continue

        # 3. Compute residuals for all point pairs under this model and count inliers
        residuals = compute_residuals(source_points, target_points, R_est, t_est, s_est)
        inliers_mask = (residuals < threshold)
        inlier_count = np.sum(inliers_mask)

        # 4. Compare and update the best model
        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            # best_model = (R_est, t_est, s_est)
            best_inliers = inliers_mask

            # If the number of inliers has reached a certain ratio, we can exit early
            if min_inlier_ratio > 0 and max_inlier_count > min_inlier_ratio * N:
                break

    # Use the best set of inliers to perform a global solve to get a more stable R, t, s
    if max_inlier_count >= 3:
        inlier_source = source_points[best_inliers]
        inlier_target = target_points[best_inliers]
        best_R, best_t, best_s = solve_method(inlier_source, inlier_target)
    else:
        # If there are not even 3 inliers, raise an exception
        raise ValueError("No inliers found in RANSAC.")

    print(f"RANSAC finished with {max_inlier_count} inliers.")
    return best_R, best_t, best_s

# ─── quick sanity check ───────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    N = 1000
    Pgen = np.random.randn(N, 3)

    # Construct random ground truth
    
    R_true  = quaternion_to_matrix_tensor(torch.tensor(np.random.randn(4))).numpy()
    S_true  = np.diag([1.2, 0.8, 1.5])
    R2_true = quaternion_to_matrix_tensor(torch.tensor(np.random.randn(4))).numpy()                    # Second rotation
    t_true  = np.array([0.3, -1.1, 2.0])

    Ppar = (R_true @ R2_true.T @ S_true @ R2_true @ Pgen.T).T + t_true + np.random.randn(N, 3) * 0.1

    R_est, t_est, S_est, R2_est = polar_12dof_np(Pgen, Ppar)

    print("R error:", np.linalg.norm(R_true - R_est))
    print("S diag :", np.diag(S_est))
    print("t error:", np.linalg.norm(t_true - t_est))