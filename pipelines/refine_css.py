import os
import math
from collections import defaultdict, OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import pickle

import utils.data as config
from networks.resnet_css import setup_css
from datasets.kitti import KITTI3D
from pipelines.detection_3d import Detection3DEvaluator, clean_kitti_data, CoordinateFrame
from pipelines.optimizer import Optimizer
from sdfrenderer.grid import Grid3D
from utils.pose import PoseEstimator
import sdfrenderer.deepsdf.workspace as dsdf_ws
import utils.refinement as rtools
import utils.visualizer as viztools

# Define seed for reproducibility
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)


def refine_css(cfgp, subset_frames=None):
    """
    Estimate and refine poses and shapes coming from the CSS network

    Args:
        cfgp: Configuration parser
        subset_frames (list): KITTI frame indices to be estimated
    """
    # Set device and precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = config.read_cfg_precision(cfgp, 'optimization', 'precision', default=torch.float16)

    # Setup CSS
    css_path = config.read_cfg_string(cfgp, 'input', 'css_path', default='')
    css_net = setup_css(pretrained=True, model_path=css_path).to(device)

    # Setup DeepSDF
    dsdf_path = config.read_cfg_string(cfgp, 'input', 'deepsdf_path', default=None)
    dsdf, latent_size = dsdf_ws.setup_dsdf(dsdf_path, precision=precision)
    dsdf = dsdf.to(device)

    # Load dataset
    data_path = config.read_cfg_string(cfgp, 'input', 'kitti_path', default=None)
    maskrcnn_labels_path = config.read_cfg_string(cfgp, 'input', 'maskrcnn_labels_path', default='')
    label_type = config.read_cfg_string(cfgp, 'input', 'label_type', default='gt')
    kitti = KITTI3D(path=data_path)

    # Throw in here all the annotations and labels we acquire over all frames
    total_annotations, total_estimations = OrderedDict(), OrderedDict()

    # Path for output autolabels
    path_autolabels = config.read_cfg_string(cfgp, 'output', 'labels', default='')
    os.makedirs(path_autolabels, exist_ok=True)

    # Check whether we only want to process a subset of frames
    if subset_frames is None:
        subset_frames = range(len(kitti))

    # Loop through frames
    for frame_idx in subset_frames:
        print('Frame', frame_idx)

        if os.path.exists(os.path.join(path_autolabels, str(frame_idx) + '.pkl')):
            print('file exists already!')
            continue

        # Fetch frame and skip if no car annotations
        sample = kitti.__getitem__(frame_idx)
        if not [a for a in sample['gt'] if a['name'] == 'Car']:
            continue

        # Build container dicts to hold annotations and labels for later evaluation
        frame_annos, frame_estimations = defaultdict(list), defaultdict(list)

        # Select annotations based on difficulty
        diff_annos = config.read_cfg_string(cfgp, 'input', 'diff_annos', default='')
        annos = rtools.get_annos(diff_annos, sample)

        # Load MaskRCNN labels, skip frame if no labels found
        if label_type != 'gt':
            maskrcnn_label_path = os.path.join(maskrcnn_labels_path, str(frame_idx)) + '.lbl'
            if os.path.exists(maskrcnn_label_path):
                maskrcnn_labels = torch.load(maskrcnn_label_path)
            else:
                print('Skip frame')
                continue

        # Loop through annotations
        for anno_idx, anno in enumerate(annos):

            # Store this annotation for later evaluation
            [frame_annos[key].append(value) for key, value in anno.items()]

            # If maskrcnn labels are available
            if label_type != 'gt':
                # Find closest maskrcnn bbox by iou
                iou = []
                for id, bbox in enumerate(maskrcnn_labels['bboxes'].numpy()):
                    iou.append(rtools.get_iou(bbox, anno['bbox']))

                bbox_max_id = np.argmax(iou)
                bbox_maskrcnn = maskrcnn_labels['bboxes'][bbox_max_id].numpy()

                # if IOU < 50, skip!
                if iou[bbox_max_id] < 0.5:
                    print('Skip frame!')
                    continue

                anno['bbox'] = bbox_maskrcnn.astype(np.int)

            # Get crops
            max_crop_area = config.read_cfg_int(cfgp, 'input', 'rendering_area', default=64) ** 2

            # Get detected crop
            l, t, r, b = anno['bbox']
            crop_bgr = sample['image'][t:b, l:r].copy()
            crop_dep = sample['depth'][t:b, l:r].copy()

            # Adjust intrinsics based on crop position and size
            K = sample['orig_cam']
            crop_size = torch.Tensor(crop_bgr.shape[:-1])
            crop_size, intrinsics, off_intrinsics = rtools.adjust_intrinsics_crop(
                K, crop_size, anno['bbox'], max_crop_area
            )
            pcd_crop, pcd_crop_rgb = rtools.reproject(crop_bgr, crop_dep, off_intrinsics, filter=False)

            # Use masks from maskrcnn
            if label_type == 'maskrcnn':
                mask = maskrcnn_labels['masks'][bbox_max_id]
                crop_bgr *= mask.unsqueeze(-1).float().expand_as(torch.tensor(crop_bgr)).numpy()

            # Preprocess image patch for pytorch digestion
            crop_rgb, crop_rgb_vis = rtools.transform_bgr_crop(crop_bgr, orig=True)
            crop_rgb = crop_rgb.unsqueeze(0).to(device).float()

            # Get css output
            pred_css = css_net(crop_rgb)
            nocs_pred = pred_css['uvw_sm_masked'].detach().squeeze() / 255.
            latent_pred = pred_css['latent'][0].detach().to(precision)

            # DeepSDF Inference and surface point/normal extraction.
            grid_density = config.read_cfg_int(cfgp, 'input', 'grid_density', default=30)
            grid = Grid3D(grid_density, device, precision)

            inputs = torch.cat([latent_pred.expand(grid.points.size(0), -1), grid.points],
                               1).to(latent_pred.device, latent_pred.dtype)
            pred_sdf_grid, inv_scale = dsdf(inputs)
            pcd_dsdf, nocs_dsdf, normals_dsdf = grid.get_surface_points(pred_sdf_grid)

            # Reproject NOCS into the scene
            nocs_pred_resized = F.interpolate(nocs_pred.unsqueeze(0), size=crop_dep.shape[:2],
                                              mode='nearest').squeeze(0)
            nocs_3d_pts, nocs_3d_cls = rtools.reproject(
                nocs_pred_resized, torch.Tensor(crop_dep).unsqueeze(0), off_intrinsics, filter=True
            )

            # Estimating initial pose
            pose_esimator_type = config.read_cfg_string(cfgp, 'optimization', 'pose_estimator', default='kabsch')
            scale = 2.0  # scale_dsdf.item() * 0.8
            pose_esimator = PoseEstimator(pose_esimator_type, scale)
            init_pose = pose_esimator.estimate(
                pcd_dsdf, nocs_dsdf, nocs_3d_pts, nocs_3d_cls, off_intrinsics, nocs_pred_resized
            )

            if init_pose is None:
                print('NO RANSAC POSE FOUND!!!')
                continue
            scale, rot, tra = init_pose['scale'], init_pose['rot'], init_pose['tra']

            # Constrain rotation to azimuth only. We need to flip X from the car system
            rot[:, 1] = [0, 1, 0]
            rot[1, :] = [0, 1, 0]
            yaw = rtools.roty_in_bev(rot @ np.diag([-1, 1, 1])) + math.pi / 2  # KITTI roty starts at canonical pi/2

            # Estimate good height by looking up lowest Y value of reprojected NOCS
            world_points = ((rot @ (pcd_dsdf.detach().cpu().numpy() * scale).T).T + tra)
            proj_world = rtools.project(sample['orig_cam'], world_points)
            L, T = proj_world[:, 0].min(), proj_world[:, 1].min()
            R, B = proj_world[:, 0].max(), proj_world[:, 1].max()
            iou = rtools.compute_iou([l, t, r, b], [L, T, R, B])
            if iou < 0.7:
                print('Restimating height')
                ymin, ymax = world_points[:, 1].min(), world_points[:, 1].max()
                tra[1] = nocs_3d_pts[:, 1].min() + (ymax - ymin) / 2

            # Optimizer and params
            params = {}
            params['yaw'] = np.array([yaw])
            params['trans'] = init_pose['tra'] / init_pose['scale']
            params['scale'] = np.array([init_pose['scale']])
            params['latent'] = latent_pred.detach().cpu().numpy()

            weights = {}
            weights['2d'] = config.read_cfg_float(cfgp, 'losses', '2d_weight', default=1)
            weights['3d'] = config.read_cfg_float(cfgp, 'losses', '3d_weight', default=1)

            # Refine the initial estimate
            optimizer = Optimizer(params, device, weights)

            # For additional visualization
            frame_vis = {}
            frame_vis['image'] = sample['image']
            frame_vis['bbox'] = anno['bbox']
            frame_vis['crop_size'] = crop_bgr.shape[:-1]

            # Set visualization type
            viz_type = config.read_cfg_string(cfgp, 'visualization', 'viz_type', default=None)

            # Optimize the initial pose estimate
            iters_optim = config.read_cfg_int(cfgp, 'optimization', 'iters', default=100)
            optimizer.optimize(
                iters_optim,
                nocs_pred,
                pcd_crop,
                dsdf,
                grid,
                intrinsics.detach().to(device, precision),
                crop_size,
                frame_vis=frame_vis,
                viz_type=viz_type
            )

            # Now collect the results from the optimization
            label_kitti, scaled_points, cam_T = rtools.get_kitti_label(dsdf, grid, params['latent'].to(precision),
                                                       params['scale'].to(precision), params['trans'].to(precision),
                                                       params['yaw'].to(precision), sample['world_to_cam'], anno['bbox'])
            [frame_estimations[key].append(value) for key, value in label_kitti.items()]

            # Visualize the estimated label in 3D
            #viztools.plot_3d_final(sample['lidar'], cam_T, scaled_points, anno, label_kitti)

        # Do not store anything for this frame if we did not process any annotations
        if not frame_annos:
            continue

        # Transform all annotations and labels into needed format and save these frame results
        necessary_keys = ['alpha', 'bbox', 'dimensions', 'location', 'rotation_y', 'score']
        for key in necessary_keys:
            frame_annos[key] = np.asarray(frame_annos[key])
            frame_estimations[key] = np.asarray(frame_estimations[key])

        # Store autolabels for later evaluation
        pickle.dump([frame_annos, frame_estimations], open(os.path.join(path_autolabels, str(frame_idx) + '.pkl'), 'wb'))
        total_annotations[frame_idx] = frame_annos
        total_estimations[frame_idx] = frame_estimations

    # Evaluation
    evaluator = Detection3DEvaluator(clean_kitti_data, compute_nuscenes=False, coordinate_frame=CoordinateFrame.CAMERA)
    formatted_result, result_dict = evaluator.evaluate_detection_3d(
        list(total_annotations.values()), list(total_estimations.values()), ['Car'], difficulties=[0]
    )
    print(formatted_result)

    evaluator = Detection3DEvaluator(clean_kitti_data, compute_nuscenes=True, coordinate_frame=CoordinateFrame.CAMERA)
    formatted_result, result_dict = evaluator.evaluate_detection_3d(
        list(total_annotations.values()), list(total_estimations.values()), ['Car'], difficulties=[0]
    )
    print(formatted_result)
