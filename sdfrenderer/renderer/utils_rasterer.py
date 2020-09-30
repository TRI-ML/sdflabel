import torch
import numpy as np
from scipy.spatial import ConvexHull


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = v.shape
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def sphericalFlip(points, center, param):
    """
    Function used to Perform Spherical Flip on the Original Point Cloud
    Source: https://github.com/williamsea/Hidden_Points_Removal_HPR.git
    """
    n = len(points)  # total n points
    points[:, 1] *= -1
    points[:, 2] *= -1
    points = points - np.repeat(center, n, axis=0)  # Move C to the origin
    normPoints = np.linalg.norm(points, axis=1)  # Normed points
    R = np.repeat(max(normPoints) * np.power(30, param), n, axis=0)  # Radius of Sphere

    flippedPointsTemp = 2 * np.multiply(np.repeat((R - normPoints).reshape(n, 1), len(points[0]), axis=1), points)
    flippedPoints = np.divide(
        flippedPointsTemp, np.repeat(normPoints.reshape(n, 1), len(points[0]), axis=1)
    )  # Apply Equation to get Flipped Points
    flippedPoints += points

    return flippedPoints


def convexHull(points):
    """
    Function used to Obtain the Convex hull
    Source: https://github.com/williamsea/Hidden_Points_Removal_HPR.git
    """
    points = np.append(points, [[0, 0, 0]], axis=0)  # All points plus origin
    hull = ConvexHull(points)  # Visible points plus possible origin. Use its vertices property.

    return hull


def calibration_matrix(resolution_px, diagonal_mm, focal_len_mm, skew=0.):
    """
    Return calibration matrix K given camera information
    Diagonal in mm of the camera sensor (ratio will match px_ratio)
    Source: https://github.com/ndrplz/differentiable-renderer/blob/pytorch/rastering/utils.py
    """
    # Camera intrinsics parameters
    resolution_x_px, resolution_y_px = resolution_px  # image resolution in pixels
    diagonal_px = np.sqrt(resolution_x_px**2 + resolution_y_px**2)

    resolution_x_mm = resolution_x_px / diagonal_px * diagonal_mm
    resolution_y_mm = resolution_y_px / diagonal_px * diagonal_mm

    skew = skew  # "skew param will be zero for most normal cameras" Hartley, Zisserman

    m_x = resolution_x_px / resolution_x_mm
    m_y = resolution_y_px / resolution_y_mm

    alpha_x = focal_len_mm * m_x  # focal length of the camera in pixels
    alpha_y = focal_len_mm * m_y  # focal length of the camera in pixels

    x_0 = resolution_x_px / 2
    y_0 = resolution_y_px / 2

    return np.array([[alpha_x, skew, x_0], [0, alpha_y, y_0], [0, 0, 1]])
