import open3d as o3d
import numpy as np
import cv2
import torch.nn.functional as F

import utils.refinement as rtools


def plot_patches(rendering_nocs, css_nocs):
    """
    Plot 2D patches: rgb patch, target nocs patch, rendering nocs patch
    Args:
        rendering_nocs (torch.Tensor): NOCS image from the renderer
        css_nocs (torch.Tensor): NOCS image prediction from the CSS network
    """
    tmp1 = rendering_nocs.clone().detach().cpu().permute(1, 2, 0).numpy()[:, :, ::-1]
    cv2.imshow('target_nocs', css_nocs.clone().detach().cpu().permute(1, 2, 0).numpy()[:, :, ::-1])
    cv2.imshow('render', tmp1.astype(np.float32))
    cv2.waitKey(10)


def plot_full_frame(frame, rendering_normals, overlay_alpha=0.1):
    """
    Visualize the full frame
    Args:
        frame: kitti frame object
        rendering_normals (torch.Tensor): Normals image from the renderer
        overlay_alpha (float): alpha overlay value defining the transparancy
    """
    # Retrieve the full image
    sample_full = frame['image'].copy()

    # Retrieve bounding box coordinates
    l, t, r, b = frame['bbox']
    cv2.rectangle(sample_full, (l, t), (r, b), (0, 0, 1), 2)

    # Get rendering mask and normals
    normals_render = (
        F.interpolate(rendering_normals.unsqueeze(0), size=frame['crop_size'],
                      mode='nearest').squeeze(0).permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]
    ).astype(np.float)
    mask = ((normals_render > 0).sum(2) > 0)

    # Overlay rendering with the full frame image
    sample_full[t:b, l:r][mask] = (overlay_alpha * sample_full[t:b, l:r] + (1 - overlay_alpha) * normals_render)[mask]

    cv2.imshow('full frame', sample_full.astype(np.float32))
    cv2.waitKey(10)


def plot_3d(viz, pcd_1, clr_1, pcd_2, clr_2, dists, idxs, interactive=False):
    """
    Visualize optimization in 3D
    Args:
        viz: Open3D visualizer object
        pcd_1 (torch.Tensor): First point cloud (N, 3)
        clr_1 (torch.Tensor): Colors of the first point cloud (N, 3)
        pcd_2 (torch.Tensor): Second point cloud (N, 3)
        clr_2 (torch.Tensor): Colors of the second point cloud (N, 3)
        dists (np.array): Distances between the correspondence points (N)
        idxs (np.array): Ids between the correspondence points (N)
        interactive (bool): activate interactive visualization mode
    """
    pcd_ren, pcd_patch = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pcd_patch.points = o3d.utility.Vector3dVector(pcd_1.detach().cpu())
    pcd_patch.colors = o3d.utility.Vector3dVector(clr_1.detach().cpu().numpy())
    pcd_ren.points = o3d.utility.Vector3dVector(pcd_2.detach().cpu().numpy())
    pcd_ren.colors = o3d.utility.Vector3dVector(clr_2.detach().cpu().numpy())
    heat_dists = rtools.build_heatmap(dists, min=0).astype(np.float32)
    line_set = rtools.build_correspondence_lineset(pcd_2.detach().cpu(), pcd_1.detach().cpu(), idxs)
    if heat_dists.size != 0:
        line_set.colors = o3d.utility.Vector3dVector(np.array(heat_dists.squeeze()))

    if interactive:
        o3d.visualization.draw_geometries([pcd_patch, pcd_ren, line_set])
    else:
        [viz.add_geometry(o) for o in [pcd_patch, pcd_ren, line_set]]
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic.set_intrinsics(800, 600, 365.4020, 365.6674, 400 - 0.5, 300 - 0.5)
        pos = np.asarray(pcd_ren.points).mean(0)
        pos -= 1.5
        params.extrinsic = rtools.lookat(pos=pos, target=np.asarray(pcd_ren.points).mean(0))
        viz.get_view_control().convert_from_pinhole_camera_parameters(params)
        viz.get_view_control().change_field_of_view(0.3)
        viz.update_geometry()
        viz.poll_events()
        viz.update_renderer()
        [viz.remove_geometry(o) for o in [pcd_patch, pcd_ren, line_set]]


def plot_3d_final(lidar, cam_T, scaled_points, est_label, gt_label=None):
    """
    Visualize the final labelling result in 3D
    Args:
        lidar (np.array): LIDAR point cloud (N,3)
        cam_T (np.array): Pose matrix (4,4)
        scaled_points (np.array): Estimated shape point cloud (N,3)
        est_label (dict): Estimated label in the KITTI format
        gt_label (dict): GT label in the KITTI format
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar)

    # Objects
    latent_pcd = o3d.geometry.PointCloud()
    latent_pcd.points = o3d.utility.Vector3dVector((cam_T[:3, :3] @ scaled_points.T).T + cam_T[:3, 3])
    colors = [[1, 0, 0] for i in range(len(latent_pcd.points))]
    latent_pcd.colors = o3d.utility.Vector3dVector(colors)

    height, width, length = est_label['dimensions']
    est_box = rtools.transform_kitti_to_cuboid(width, height, length, est_label['location'], est_label['rotation_y'])
    est_box = rtools.build_vizbox(est_box, [0, 0, 1])
    origin_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)

    if gt_label:
        gt_dims, gt_loc, gt_rot = gt_label['dimensions'], gt_label['location'], gt_label['rotation_y']
        gt_box = rtools.transform_kitti_to_cuboid(gt_dims[1], gt_dims[0], gt_dims[2], gt_loc, gt_rot)
        gt_box = rtools.build_vizbox(gt_box, [0, 1, 0])

    o3d.visualization.draw_geometries([pcd, est_box, latent_pcd, gt_box, origin_mesh])
