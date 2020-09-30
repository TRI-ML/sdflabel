import math
import torch
import numpy as np
from renderer.utils_rasterer import convexHull, sphericalFlip, qrot


def project_in_2D(
    K, camera_pose, points, normals, colors, resolution_px, filter_normals=True, filter_hpr=False, output_nocs=True
):
    """
    Project all 3D points onto the 2D image of given resolution using DCM rotation matrix

    Args:
        K (torch.Tensor): Intrinsic camera parameters (3,3)
        camera_pose (torch.Tensor): Camera pose as DCM (4,4)
        points (torch.Tensor): Object points to project (N,3)
        normals (torch.Tensor): Normals per point (N,3)
        colors (torch.Tensor): Point colors (N,3)
        resolution_px (tuple): Screen resolution
        filter_normals (bool): Filter points based on the normals
        filter_hpr (bool): Filter points based on the hpr filter
        output_nocs (bool): Output NOCS
    """
    resolution_x_px, resolution_y_px = resolution_px  # image resolution in pixels

    # Define precision and device
    eps = torch.finfo(K.dtype).eps
    device = K.device
    dtype = K.dtype

    # Ouput dict
    output = {}

    RT = (camera_pose[:-1, :]).to(device, dtype)  # remove homogeneous row

    # Correct reference system of extrinsics matrix
    #   y is down: (to align to the actual pixel coordinates used in digital images)
    #   right-handed: positive z look-at direction
    correction_factor = torch.from_numpy(np.asarray([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                                                    dtype=np.float32)).to(device, dtype)
    RT = correction_factor @ RT

    # Create constant tensor to store 3D model coordinates
    ones = torch.ones(points[:, :1].shape).to(device, dtype)
    coords_3d_h = torch.cat([points, ones], dim=-1)  # n_vertices, 4
    coords_3d_h = coords_3d_h.t()  # 4, n_vertices

    # Project normals
    normals_projected = (RT[:, :3] @ normals.t()).t()

    # Fake colors: coordinates
    # colors = (RT[:, :3] @ mesh.t()).t()
    if output_nocs:
        colors = points.clone()
        colors[:, 0] *= -1

    # Project points and filter where normals point away from the camera
    coords_projected_3d = (RT @ coords_3d_h).t()

    # Filter invisible points
    if filter_normals:
        dot_prod = torch.bmm(normals_projected.unsqueeze((-2)), coords_projected_3d.unsqueeze(-1)).squeeze(-1)
        # Filter where normals point away from the camera
        coords_projected_3d_filt = coords_projected_3d.masked_select(dot_prod < 0).view(-1, 3)
        colors_filt = colors.masked_select(dot_prod < 0).view(-1, 3)
        normals_projected_filt = normals_projected.masked_select(dot_prod < 0).view(-1, 3)

        output['points_3d_filt'] = coords_projected_3d_filt
        output['normals_3d_filt'] = normals_projected_filt
        output['colors_3d_filt'] = colors_filt

    if filter_hpr:
        # Filter based on HPR operator
        C = np.array([[0, 0, 0]])  # center point
        coords_projected_3d_numpy = coords_projected_3d.detach().cpu().numpy()
        coords_projected_3d_numpy /= coords_projected_3d_numpy.max()
        flippedPoints = sphericalFlip(coords_projected_3d_numpy, C, math.pi)
        mask_ids = convexHull(flippedPoints).vertices[:-1]
        mask = np.zeros_like(coords_projected_3d_numpy[:, 2:])
        mask[mask_ids] = 1
        mask = torch.BoolTensor(mask).to(device)

        coords_projected_3d = coords_projected_3d.masked_select(mask).view(-1, 3)
        colors = colors.masked_select(mask).view(-1, 3)
        normals_projected = normals_projected.masked_select(mask).view(-1, 3)

    # Project 3D vertices into 2D
    coords_projected_2d_h = (K @ coords_projected_3d.t()).t()
    coords_projected_2d = coords_projected_2d_h[:, :2] / (coords_projected_2d_h[:, 2:] + eps)

    # Clip indexes in image range (off by 1 pixel each side to avoid edge issues)
    coords_projected_2d_x_clip = torch.clamp(coords_projected_2d[:, 0:1], -1, resolution_x_px)
    coords_projected_2d_y_clip = torch.clamp(coords_projected_2d[:, 1:2], -1, resolution_y_px)

    # Fill a dictionary
    output['points_3d'] = coords_projected_3d
    output['normals_3d'] = normals_projected
    output['colors_3d'] = colors
    output['points_2d'] = torch.cat([coords_projected_2d_x_clip, coords_projected_2d_y_clip], dim=-1)

    return output


def project_in_2D_quat(
    K, camera_pose, points, normals, colors, resolution_px, filter_normals=False, filter_hpr=False, output_nocs=True
):
    """
    Project all 3D points onto the 2D image of given resolution using quaternions

    Args:
        K (torch.Tensor): Intrinsic camera parameters (3,3)
        camera_pose (torch.Tensor): Camera pose as quaternion [:4] and translation vector [4:]
        points (torch.Tensor): Object points to project (N,3)
        normals (torch.Tensor): Normals per point (N,3)
        colors (torch.Tensor): Point colors (N,3)
        resolution_px (tuple): Screen resolution
        filter_normals (bool): Filter points based on the normals
        filter_hpr (bool): Filter points based on the hpr filter
        output_nocs (bool): Output NOCS
    """
    resolution_x_px, resolution_y_px = resolution_px  # image resolution in pixels

    # Define precision and device
    eps = torch.finfo(K.dtype).eps
    device = K.device
    dtype = K.dtype

    # Ouput dict
    output = {}

    q = camera_pose[:4]
    t = camera_pose[4:]

    # Correct reference system of extrinsics matrix
    #   y is down: (to align to the actual pixel coordinates used in digital images)
    #   right-handed: positive z look-at direction
    correction_factor = torch.from_numpy(np.asarray([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                                                    dtype=np.float32)).to(device, dtype)
    correction_factor.requires_grad = True

    # Project normals
    normals_projected = qrot(q.unsqueeze(0).expand([normals.size(0), 4]), normals)
    normals_projected = (correction_factor @ normals_projected.t()).t()

    # Fake colors: coordinates
    # colors = qrot(q.unsqueeze(0).expand([coords.size(0), 4]), coords)
    if output_nocs:
        colors = points.clone()
        # colors[:, 0] *= -1

    # Project points
    coords_rotated_3d_quat = qrot(q.unsqueeze(0).expand([points.size(0), 4]), points)
    corrT = correction_factor @ torch.cat([torch.eye(3).to(device, dtype), t.unsqueeze(-1)], dim=-1)
    coords_projected_3d = (
        corrT
        @ torch.cat([coords_rotated_3d_quat, torch.ones(points[:, :1].shape).to(device, dtype)], dim=-1).t()
    ).t()

    # Filter invisible points
    if filter_normals:
        dot_prod = torch.bmm(normals_projected.unsqueeze((-2)), coords_projected_3d.unsqueeze(-1)).squeeze(-1)
        # Filter where normals point away from the camera
        coords_projected_3d_filt = coords_projected_3d.masked_select(dot_prod < 0).view(-1, 3)
        colors_filt = colors.masked_select(dot_prod < 0).view(-1, 3)
        normals_projected_filt = normals_projected.masked_select(dot_prod < 0).view(-1, 3)

        output['points_3d_filt'] = coords_projected_3d_filt
        output['normals_3d_filt'] = normals_projected_filt
        output['colors_3d_filt'] = colors_filt

    elif filter_hpr:
        # Filter based on HPR operator
        C = np.array([[0, 0, 0]])  # center point
        coords_projected_3d_numpy = coords_projected_3d.detach().cpu().numpy()
        flippedPoints = sphericalFlip(coords_projected_3d_numpy, C, math.pi)
        mask_ids = convexHull(flippedPoints).vertices[:-1]
        mask = np.zeros_like(coords_projected_3d_numpy[:, 2:])
        mask[mask_ids] = 1
        mask = torch.BoolTensor(mask).to(device)

        coords_projected_3d = coords_projected_3d.masked_select(mask).view(-1, 3)
        colors = colors.masked_select(mask).view(-1, 3)
        normals_projected = normals_projected.masked_select(mask).view(-1, 3)

    # Project 3D vertices into 2D
    coords_projected_2d_h = (K @ coords_projected_3d.t()).t()
    coords_projected_2d = coords_projected_2d_h[:, :2] / (coords_projected_2d_h[:, 2:] + eps)

    # Clip indexes in image range (off by 1 pixel each side to avoid edge issues)
    coords_projected_2d_x_clip = torch.clamp(coords_projected_2d[:, 0:1], -1, resolution_x_px)
    coords_projected_2d_y_clip = torch.clamp(coords_projected_2d[:, 1:2], -1, resolution_y_px)

    # Fill a dictionary
    output['points_3d'] = coords_projected_3d
    output['normals_3d'] = normals_projected
    output['colors_3d'] = colors
    output['points_2d'] = torch.cat([coords_projected_2d_x_clip, coords_projected_2d_y_clip], dim=-1)

    return output
