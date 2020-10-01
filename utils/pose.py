import torch
import numpy as np
from sklearn.neighbors import KDTree
import cv2
from scipy.spatial.transform import Rotation as R


class PoseEstimator:
    def __init__(self, type='kabsch', scale=2.2):
        self.scale = scale
        self.type = type

    def estimate(self, pcd_dsdf, nocs_dsdf, pcd_scene, nocs_scene, off_intrinsics, nocs_pred_resized):
        """
        Estimate the pose based on the pose estimator type
        Args:
            pcd_dsdf (torch.Tensor): DeepSDF points cloud (N,3)
            nocs_dsdf (torch.Tensor): DeepSDF NOCS (N,3)
            pcd_scene (torch.Tensor): Scene point cloud(N,3)
            nocs_scene (torch.Tensor): Scene NOCS (N,3)
            off_intrinsics (torch.Tensor): Camera intrinsic matrix (3,3)
            nocs_pred_resized (torch.Tensor): NOCS image

        Returns: Pose dictionary

        """
        if self.type == 'kabsch':
            init_pose = self.init_pose_3d(
                pcd_dsdf, nocs_dsdf, pcd_scene, nocs_scene, type='kabsch', scale_model=self.scale
            )
        elif self.type == 'procrustes':
            init_pose = self.init_pose_3d(
                pcd_dsdf, nocs_dsdf, pcd_scene, nocs_scene, type='procrustes', scale_model=self.scale
            )
        elif self.type == 'pnp':
            init_pose = self.init_pose_2d(off_intrinsics, nocs_pred_resized, scale_model=self.scale)

        return init_pose

    @staticmethod
    def init_pose_2d(cam, nocs_region, scale_model=1):
        """
        PnP based pose estimation
        Args:
            cam (torch.Tensor): Intrinsic camera matrix (3,3)
            nocs_region (torch.Tensor): NOCS image
            scale_model (float): Scale of the estimated object

        Returns: Pose dictionary

        """
        nocs_region_np = nocs_region.detach().cpu().permute(1, 2, 0).numpy()
        nonzero_mask = nocs_region_np[:, :, 0] > 0
        nocs_values = nocs_region_np[nonzero_mask]
        points_3d = (nocs_values * 2) - 1

        grid_row, grid_column = np.nonzero(nonzero_mask)

        image_points = np.empty((len(grid_row), 2))
        image_points[:, 0] = grid_row
        image_points[:, 1] = grid_column

        object_points = points_3d
        object_points *= scale_model

        predicted_pose = solvePnP(cam.cpu().numpy(), image_points, object_points)

        # Convert to our format
        rot = predicted_pose[:3, :3]
        quat = R.from_dcm(rot).as_quat()
        quat = np.concatenate([quat[3:], quat[:3]])  # reformat
        trans = predicted_pose[:3, 3]

        # Write pose
        pose = {}
        pose['rot'] = rot
        pose['quat'] = quat
        pose['tra'] = trans
        pose['scale'] = scale_model

        return pose

    @staticmethod
    def init_pose_3d(
        model_pts,
        model_cls,
        scene_pts,
        scene_cls,
        metric_distance_threshold=0.15,
        nocs_distance_threshold=0.15,
        type='procrustes',
        scale_model=1
    ):
        """
        Kabsch/Procrustes-based pose estimation
        Args:
            model_pts (np.array): Models points
            model_cls (np.array): Models colors
            scene_pts (np.array): Scene points
            scene_cls (np.array): Scene colors
            metric_distance_threshold (float):
            nocs_distance_threshold (float):
            type ('kabsch'/'procrustes'): Kabsch or Procrustes
            scale_model (float): Scale of the estimated object

        Returns: Pose dictionary

        """
        if isinstance(scene_pts, torch.Tensor):
            scene_pts = scene_pts.detach().cpu().numpy()
        if isinstance(scene_cls, torch.Tensor):
            scene_cls = scene_cls.detach().cpu().numpy()
        if isinstance(model_pts, torch.Tensor):
            model_pts = model_pts.detach().cpu().numpy()
        if isinstance(model_cls, torch.Tensor):
            model_cls = model_cls.detach().cpu().numpy()

        if scene_pts.shape[0] < 5:
            return None

        if type == 'kabsch':
            model_pts *= scale_model

        total_num_points = scene_pts.shape[0]

        # Optimization parameters
        p = 0.99
        outlier_prob = 0.7
        ransac_sample_size = 4
        num_ransac_iterations = int(
            round((np.log(1.0 - p) / np.log(1 - pow(1 - outlier_prob, ransac_sample_size))) + 0.5)
        )
        min_num_inliers = 5
        best_inlier_indices = np.array([])

        kdtree_colors = KDTree(model_cls)
        kdtree_points = KDTree(model_pts)

        for _ in range(num_ransac_iterations):

            # Take N random samples from the scene
            indices = np.random.choice(range(total_num_points), ransac_sample_size, replace=False)
            selected_scene_pts = scene_pts[indices]
            selected_scene_cls = scene_cls[indices]

            # Find closest NOCS correspondences in the model
            dists, idxs_nocs = kdtree_colors.query(selected_scene_cls)
            idxs_nocs = [val for sublist in idxs_nocs for val in sublist]
            dists = np.asarray([val for sublist in dists for val in sublist])

            # If not color-compatible, try next samples
            if (dists > nocs_distance_threshold).any():
                continue

            selected_model_pts = model_pts[idxs_nocs]
            selected_model_cls = model_cls[idxs_nocs]

            # Compute transform of the 3D points
            if type == 'procrustes':
                result = procrustes(selected_scene_pts, selected_model_pts)
                if result is None:
                    continue
                scale, rot, tra = result
            elif type == 'kabsch':
                rot, tra = kabsch(selected_scene_pts, selected_model_pts)
                scale = 1

            if scale > 3:
                continue

            trans = np.zeros((3, 4), dtype=np.float32)
            trans[:3, :3] = rot * scale
            trans[:3, 3] = tra
            transformed_scene = (trans[:, :3] @ scene_pts.T).T + trans[:, 3]

            # Compute for each model point the closest scene point in 3D space
            dists, idxs = kdtree_points.query(transformed_scene)
            idxs = [val for sublist in idxs for val in sublist]
            dists = np.asarray([val for sublist in dists for val in sublist])
            dists_color = np.linalg.norm(scene_cls - model_cls[idxs], axis=1)

            # Check how many of those are below distance treshold, i.e. inliers
            inlier_indices = np.where((dists < metric_distance_threshold) & (dists_color < nocs_distance_threshold))[0]
            # print(len(inlier_indices), scale, trans)

            # 3D plot of 3D render, patch and correspondences
            if False:
                pcd_scene, pcd_model = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
                pcd_scene.points = o3d.utility.Vector3dVector(scene_pts)
                pcd_scene.colors = o3d.utility.Vector3dVector(scene_cls)
                pcd_model.points = o3d.utility.Vector3dVector(model_pts)
                pcd_model.colors = o3d.utility.Vector3dVector(model_cls)
                line_set = build_correspondence_lineset(selected_model_pts, selected_scene_pts, range(len(indices)))
                o3d.visualization.draw_geometries([pcd_scene, pcd_model, line_set])

            if len(inlier_indices) > len(best_inlier_indices):
                best_inlier_indices = inlier_indices

        # Undefined pose in bad case
        if len(best_inlier_indices) < min_num_inliers:
            return None

        selected_scene_pts = scene_pts[best_inlier_indices]
        selected_scene_cls = scene_cls[best_inlier_indices]

        dists, idxs = kdtree_colors.query(selected_scene_cls)
        idxs = [val for sublist in idxs for val in sublist]
        selected_model_pts = model_pts[idxs]
        selected_model_cls = model_cls[idxs]

        if type == 'procrustes':
            scale, rot, tra = procrustes(selected_model_pts, selected_scene_pts)
        elif type == 'kabsch':
            rot, tra = kabsch(selected_model_pts, selected_scene_pts)
            scale = scale_model

        if False:
            pcd_scene, pcd_model = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
            pcd_scene.points = o3d.utility.Vector3dVector(scene_pts)
            pcd_scene.colors = o3d.utility.Vector3dVector(scene_cls)
            pcd_model.points = o3d.utility.Vector3dVector(model_pts)
            pcd_model.colors = o3d.utility.Vector3dVector(model_cls)
            line_set = build_correspondence_lineset(
                selected_model_pts, selected_scene_pts, range(len(best_inlier_indices))
            )
            o3d.visualization.draw_geometries([pcd_scene, pcd_model, line_set])

        # Write pose
        pose = {}
        pose['scale'] = scale
        pose['rot'] = rot
        pose['tra'] = tra
        return pose


def solvePnP(cam, image_points, object_points, return_inliers=False):
    """
    OpenCV PnP Solver
    Args:
        cam (np.array): Intrinsic camera matrix (3,3)
        image_points (np.array): 2D correspondences (N,2)
        object_points (np.array): 3D correspondences (N,3)
        return_inliers (bool): Return RANSAC inliers

    Returns: Pose dictionary

    """
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    if image_points.shape[0] < 4:
        pose = np.eye(4)
        inliers = []
    else:
        image_points[:, [0, 1]] = image_points[:, [1, 0]]
        object_points = np.expand_dims(object_points, 1)
        image_points = np.expand_dims(image_points, 1)

        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            object_points,
            image_points.astype(float),
            cam,
            dist_coeffs,
            iterationsCount=1000,
            reprojectionError=1.
        )[:4]

        # Get a rotation matrix
        pose = np.eye(4)
        if success:
            pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            pose[:3, 3] = np.squeeze(translation_vector)

        if inliers is None:
            inliers = []

    if return_inliers:
        return pose, len(inliers)
    else:
        return pose


def procrustes(from_points, to_points):
    """
    Implementation of the Procrustes step
    Args:
        from_points (np.array): (4,3)
        to_points (np.array): (4,3)

    Returns: Scale (float), Rotation (3,3), Translation (3)

    """
    assert len(from_points.shape) == 2, \
        "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"

    N, m = from_points.shape

    mean_from = from_points.mean(axis=0)
    mean_to = to_points.mean(axis=0)

    delta_from = from_points - mean_from  # N x m
    delta_to = to_points - mean_to  # N x m

    sigma_from = (delta_from * delta_from).sum(axis=1).mean()
    sigma_to = (delta_to * delta_to).sum(axis=1).mean()

    cov_matrix = delta_to.T.dot(delta_from) / N

    U, d, V_t = np.linalg.svd(cov_matrix, full_matrices=True)
    cov_rank = np.linalg.matrix_rank(cov_matrix)
    S = np.eye(m)

    if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
        S[m - 1, m - 1] = -1
    elif cov_rank < m - 1:
        return None
        # raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))

    R = U.dot(S).dot(V_t)
    c = (d * S.diagonal()).sum() / sigma_from
    t = mean_to - c * R.dot(mean_from)
    return c, R, t


def kabsch(canonical_points, predicted_points):
    """
    Implementation of the Kabsch step
    Args:
        canonical_points (np.array): (4,3)
        predicted_points (np.array): (4,3)

    Returns: Rotation (3,3) and translation (3)

    """
    canonical_mean = np.mean(canonical_points, axis=0)
    predicted_mean = np.mean(predicted_points, axis=0)

    canonical_centered = canonical_points - np.expand_dims(canonical_mean, axis=0)
    predicted_centered = predicted_points - np.expand_dims(predicted_mean, axis=0)

    cross_correlation = predicted_centered.T @ canonical_centered

    u, s, vt = np.linalg.svd(cross_correlation)

    rotation = u @ vt

    det = np.linalg.det(rotation)

    if det < 0.0:
        vt[-1, :] *= -1.0
        rotation = np.dot(u, vt)

    translation = predicted_mean - canonical_mean
    translation = np.dot(rotation, translation) - np.dot(rotation, predicted_mean) + predicted_mean

    return rotation, translation
