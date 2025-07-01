from typing import Literal
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial import QhullError, ConvexHull, Delaunay
from tqdm import tqdm
import open3d as o3d

from utils.geometry import quaternion_to_matrix_np

np.random.seed(2025)

def __centroid_of_tetrahedron(vertices):
    """Compute the centroid of a tetrahedron."""
    return np.mean(vertices, axis=0)

def __volume_of_tetrahedron(vertices):
    """Compute the volume of a tetrahedron."""
    # Use determinant to compute the volume
    matrix = np.vstack([vertices.T, np.ones(4)])
    return abs(np.linalg.det(matrix)) / 6.0

def __centroid_of_convex_hull(points: np.ndarray):
    """Compute the centroid of the convex hull of a 3D point set."""
    try:
        hull = ConvexHull(points)
    except QhullError:
        print("QhullError: Could not construct convex hull, possibly due to coplanar or collinear points.")
        # In this case, consider other methods (e.g., compute in-plane if coplanar)
        return None
    # Use Delaunay triangulation to decompose the convex hull into tetrahedra
    tri = Delaunay(hull.points[hull.vertices])

    total_volume = 0
    weighted_centroid_sum = np.zeros(3)

    for simplex in tri.simplices:
        tetrahedron_vertices = hull.points[hull.vertices][simplex]
        tetrahedron_volume = __volume_of_tetrahedron(tetrahedron_vertices)
        tetrahedron_centroid = __centroid_of_tetrahedron(tetrahedron_vertices)

        total_volume += tetrahedron_volume
        weighted_centroid_sum += tetrahedron_volume * tetrahedron_centroid

    if total_volume == 0:
        # Convex hull has zero volume, possibly because all points are collinear or coplanar, returning the mean of the points.
        return np.mean(points, axis=0)

    return weighted_centroid_sum / total_volume

def get_centroid(points: np.ndarray, method: Literal["convex_hull", "bbox", "mean"] = "convex_hull"):
    """
    Compute the centroid of a 3D point set.

    Args:
        points: A numpy array of shape (N, 3) containing N 3D points.
        method: Method to compute the centroid. Options: 'convex_hull', 'bbox', 'mean'.
                'convex_hull': Use the centroid of the convex hull.
                'bbox': Use the center of the bounding box.
                'mean': Use the mean of the points.

    Returns:
        A numpy array of shape (3,) containing the centroid coordinates.
    """
    if method == 'convex_hull':
        return __centroid_of_convex_hull(points)
    elif method == 'bbox':
        return np.min(points, axis=0) + np.max(points, axis=0) / 2
    elif method == 'mean':
        return np.mean(points, axis=0)
    else:
        raise ValueError("Invalid method. Must be one of 'convex_hull', 'bbox', or 'mean'.")


def filter_outliers(
    ratio_records,
    method: Literal["iqr", "std", "manual"] = "iqr",
    threshold=1.5,
    lower_bound=None,
    upper_bound=None,
):
    """
    Filters outliers from a list of ratio records.

    Args:
        ratio_records: A list of numerical values.
        method: The method to use for outlier detection.  Options: 'iqr', 'std', 'manual'.
                'iqr':  Interquartile Range method.
                'std': Standard Deviation method.
                'manual':  Uses provided lower and upper bounds.
        threshold:  For 'iqr' and 'std' methods, this is the multiplier used to determine the outlier range.
                    (e.g., for IQR, values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR are outliers).
        lower_bound:  For 'manual' method, the lower bound for acceptable values.
        upper_bound:  For 'manual' method, the upper bound for acceptable values.

    Returns:
        A tuple containing:
            - A list of filtered values (outliers removed).
            - A list of outlier values.
            - The lower and upper bounds used for filtering.
    """
    ratio_records = np.array(ratio_records)  # Convert to NumPy array for easier calculations

    if method == 'iqr':
        Q1 = np.percentile(ratio_records, 25)
        Q3 = np.percentile(ratio_records, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

    elif method == 'std':
        mean = np.mean(ratio_records)
        std = np.std(ratio_records)
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

    elif method == 'manual':
        if lower_bound is None or upper_bound is None:
            raise ValueError("For 'manual' method, both lower_bound and upper_bound must be provided.")
    else:
        raise ValueError("Invalid method. Must be one of 'iqr', 'std', or 'manual'.")

    filtered_records = ratio_records[(ratio_records >= lower_bound) & (ratio_records <= upper_bound)]
    outliers = ratio_records[(ratio_records < lower_bound) | (ratio_records > upper_bound)]

    return filtered_records, outliers, lower_bound, upper_bound


def __save_points_to_ply(points, filename):
    """
    Save point cloud data as PLY format using plyfile library.
    
    Args:
        points: numpy array of shape (N, 3)
        filename: file path to save
    """
    # Create a structured array
    vertex = np.zeros(len(points), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]

    # Create a PlyElement object
    vertex_element = PlyElement.describe(vertex, 'vertex')

    # Create a PlyData object and write to file
    PlyData([vertex_element], text=True).write(filename)

@torch.no_grad()
def get_incremental_rotation_matrices(
    n: int,
    min_angle_diff_init: float,
    min_angle_diff_util: float,
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """
    Incrementally generate rotation matrices, ensuring that the angle difference between any two matrices is greater than a threshold.
    Use Fibonacci sphere for more uniform initial sampling.

    Args:
        n: Number of rotation matrices to generate.
        min_angle_diff_init: Initial minimum angle difference threshold.
        min_angle_diff_util: Final minimum angle difference threshold.

    Returns:
        A numpy array of shape (n, 3, 3) containing n rotation matrices.
    """

    def generate_rotation_matrices_quaternion(n):
        """
        Generate n uniformly distributed rotation matrices using quaternions.

        Args:
          n: The number of rotation matrices to generate.

        Returns:
          A numpy array of shape (n, 3, 3) containing n rotation matrices.
        """

        # 1. Uniformly sample on the 4D unit sphere
        u = np.random.normal(size=(n, 4))
        u = u / np.linalg.norm(u, axis=1)[:, np.newaxis]  # Normalize

        # 2. Convert quaternions to rotation matrices
        rotations = quaternion_to_matrix_np(u)

        return np.array(rotations)


    # # 1. Oversample to generate initial points
    # num_initial_points = n * 128
    # points = fibonacci_sphere(num_initial_points)

    # # 2. Create rotation matrices
    # z_axis = np.array([0, 0, 1])
    # initial_rotations = [rotation_from_two_vectors(z_axis, p) for p in tqdm(points)]

    # 1. Generate initial rotation matrices
    initial_rotations = generate_rotation_matrices_quaternion(n * 128).tolist()

    # 3. Create angle difference schedule
    min_angle_diff_schedule = np.linspace(min_angle_diff_init, min_angle_diff_util, n, endpoint=True)

    # 4. Iteratively select rotation matrices
    rotations = [initial_rotations.pop(0)]  # Take the first rotation matrix

    while len(rotations) < n:
        min_angle_diff = min_angle_diff_schedule[len(rotations)]
        existing = np.stack(rotations)  # (k, 3, 3)
        # Calculate the angle difference between all candidates in initial_rotations and the selected matrices at once
        cand = np.stack(initial_rotations)  # (m, 3, 3)
        traces = torch.einsum(
            "kab,mab->km",
            torch.tensor(existing, device=device, dtype=torch.float32),
            torch.tensor(cand, device=device, dtype=torch.float32),
        )
        angles = torch.arccos(torch.clip((traces - 1) / 2, -1, 1))  # (k, m)
        angles = angles.cpu().numpy()
        min_diffs = angles.min(axis=0) # Minimum angle difference for each candidate
        # print(min_diffs)
        best_idx = np.argmax(min_diffs)
        max_min_diff = min_diffs[best_idx]

        # if max_min_diff < min_angle_diff:
        #     print("Warning: Could not find a suitable rotation. Stopping early.")
        #     break

        rotations.append(initial_rotations.pop(best_idx))
        print(f"Generated {len(rotations)} rotations", end='\r')

    print(f"minimum angle difference: {max_min_diff/np.pi*180:.4f} degrees")

    return np.stack(rotations)


def image_depth2pcd(
    image: np.ndarray,
    depth: np.ndarray,
    fx,
    fy,
    cx,
    cy,
    save_path: str,
):
    """
    Project an image and depth map to a point cloud and save it as a PLY file.
    Args:
        image: RGB image with shape (3, H, W).
        depth: Depth map with shape (1, H, W).
        fx: Focal length in x direction.
        fy: Focal length in y direction.
        cx: Principal point x coordinate.
        cy: Principal point y coordinate.
        save_path: Path to save the point cloud as a PLY file.
    """
    mask = depth > 0
    v, u = np.where(mask.squeeze())
    coords = np.stack([u, v], axis=-1)
    depthes = depth[0, v, u]

    points = pix2pcd(coords, depthes, fx, fy, cx, cy)
    colors = image[:, v, u].T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(save_path, pcd)


def pix2pcd_tensor(
    coords,
    depthes,
    fx,
    fy,
    cx,
    cy,
    save_path: str = None,
    device="cuda",
) -> torch.Tensor:
    '''
    Convert pixel coordinates and depth map to point cloud coordinates.

    Args:
        coords: Pixel coordinates, shape (N, 2)
        depthes: Depth map, shape (N,)
        fx: Focal length in x direction
        fy: Focal length in y direction
        cx: Principal point x coordinate
        cy: Principal point y coordinate
    Returns: 
        Point cloud coordinates, shape (N, 3)
    '''
    u = torch.tensor(coords[:, 0], device=device)
    v = torch.tensor(coords[:, 1], device=device)
    z = torch.tensor(depthes, device=device)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = torch.stack([x, y, z], dim=-1)

    if save_path is not None:
        __save_points_to_ply(points, save_path)

    return points


def pix2pcd(coords, depthes, fx, fy, cx, cy, save_path: str = None) -> np.ndarray:
    '''
    Convert pixel coordinates and depth map to point cloud coordinates.

    Args:
        coords: Pixel coordinates, shape (N, 2)
        depthes: Depth map, shape (N,)
        fx: Focal length in x direction
        fy: Focal length in y direction
        cx: Principal point x coordinate
        cy: Principal point y coordinate
    Returns: 
        Point cloud coordinates, shape (N, 3)
    '''
    u = coords[:, 0]
    v = coords[:, 1]
    z = depthes
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)

    if save_path is not None:
        __save_points_to_ply(points, save_path)

    return points


def test():
    # Generate more rotation matrices for better visualization of spherical distribution
    n_rotations = 128
    rotations = get_incremental_rotation_matrices(n_rotations, np.pi / 3, np.pi / 3)

    point = np.array([0, 0, 1])

    # Store all rotated points
    rotated_points = [rot_matrix @ point for rot_matrix in rotations]

    # Convert to numpy array
    rotated_points = np.array(rotated_points)

    # Save result as PLY file
    __save_points_to_ply(rotated_points, "./rotation_matrices/rotated_points.ply")

    print(f"Generated {n_rotations} rotated points and saved to rotated_points.ply")
    print("Points shape:", rotated_points.shape)

    # Compute some statistics to verify the uniformity of rotations
    mean_position = np.mean(rotated_points, axis=0)
    std_position = np.std(rotated_points, axis=0)
    print("Mean position:", mean_position)
    print("Standard deviation:", std_position)

    # Save rotation matrices
    np.savez_compressed(f"./rotation_matrices/rotations_{n_rotations}_.npz", rotations=rotations)


if __name__ == "__main__":
    test()
