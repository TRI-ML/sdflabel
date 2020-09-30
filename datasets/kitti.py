import os
import numpy as np
import open3d as o3d
import cv2
from torch.utils.data import Dataset
from collections import OrderedDict
cv2.setNumThreads(0)

from utils.refinement import is_anno_easy, is_anno_moderate, compute_depth_map, reproject, build_view_frustum


def get_kitti_frame(sample):
    H, W, _ = sample['image'].shape
    # Filter out lidar points outside field of view
    scene_lidar = sample['lidar']
    frustum = build_view_frustum(sample['orig_cam'], 0, 0, W, H)
    scene_lidar = scene_lidar[np.logical_and.reduce(frustum @ scene_lidar.T > 0, axis=0)]

    # Build Open3D pcd and estimate normals
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_lidar)
    scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    # origin_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)

    # Estimate road plane (stupidly by checking orthogonality to camera. RANSAC better)
    normals = np.asarray(scene_pcd.normals)
    ortho_to_cam = np.abs(normals @ np.asarray([0, 1, 0])) > 0.9
    plane_points = scene_lidar[ortho_to_cam]
    plane_normal = np.mean(normals[ortho_to_cam], axis=0)
    plane_normal /= np.linalg.norm(plane_normal)
    plane_dists = plane_normal @ plane_points.T
    plane_offset = np.median(plane_dists)

    # Filter out road plane by simple normal check
    scene_lidar = scene_lidar[~ortho_to_cam]
    scene_pcd.points = o3d.utility.Vector3dVector(scene_lidar)
    scene_pcd.normals = o3d.utility.Vector3dVector(normals[~ortho_to_cam])
    # o3d.visualization.draw_geometries([scene_pcd, origin_mesh])

    # Compute depth map for whole image
    scene_depth = compute_depth_map(scene_lidar, sample['orig_cam'], W, H)

    # Reproject all visible, colored scene points
    pts_scene, clrs_scene = reproject(sample['image'], scene_depth, sample['orig_cam'])
    pcd = o3d.geometry.PointCloud()
    pcd.points, pcd.colors = o3d.utility.Vector3dVector(pts_scene), o3d.utility.Vector3dVector(clrs_scene)
    return scene_depth, pcd


class KITTI3D(Dataset):
    def __init__(
        self,
        path,
        training=True,
        data_split='trainval',
        debug=False,
    ):

        self.path = path
        self.train = training
        self.data_split = data_split
        self.debug = debug

        # Get file names to load from data split file
        # Split source: https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
        assert data_split in ['test', 'train', 'trainval', 'val']
        with open(os.path.join(self.path, 'mv3d_kitti_split', data_split + '.txt')) as _f:
            self.names = [line.strip() for line in _f]
            if data_split == 'test':
                self.root = os.path.join(path, 'testing')
            else:
                self.root = os.path.join(path, 'training')

        # Get relative label paths
        self.images = ['image_2/' + name + '.png' for name in self.names]
        self.label_files = ['label_2/' + name + '.txt' for name in self.names]
        self.calibs = ['calib/' + name + '.txt' for name in self.names]
        self.lidars = ['velodyne/' + name + '.bin' for name in self.names]

        depth_prefix = 'lidar_depth'
        os.makedirs(os.path.join(path, 'lidar_depth'), exist_ok=True)

        if data_split == 'test':
            self.depths = [depth_prefix + '/test_' + name + '.npz' for name in self.names]
        else:
            self.depths = [depth_prefix + '/train_' + name + '.npz' for name in self.names]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # Start building sample
        sample = OrderedDict()
        sample['idx'] = index
        sample['scale'] = 1
        sample['name'] = self.names[index]

        # Read calibration data and break apart into separate entities
        calib = open(os.path.join(self.root, self.calibs[index])).readlines()
        calib = [c[:-1].split(' ') for c in calib]

        # Parse left camera projection matrix into 3x4 form
        P2 = np.asarray([float(f) for f in calib[2][1:]]).reshape((3, 4))

        # Parse velodyne to left image transform
        velo_to_cam = np.asarray([float(f) for f in calib[5][1:]]).reshape((3, 4))

        # Reshape LIDAR data into (x, y, z, intensity) and bring into camera frame
        velodyne = np.fromfile(os.path.join(self.root, self.lidars[index]), np.float32)
        velodyne = velodyne.reshape((-1, 4))[:, :3]
        sample['lidar'] = (velo_to_cam[:3, :3] @ velodyne.T).T + velo_to_cam[:3, 3]

        # Read the image and label files
        img = cv2.imread(os.path.join(self.root, self.images[index]), -1)
        H, W, C = img.shape
        sample['image'] = img.astype(np.float32) / 255.0
        sample['orig_hw'] = (H, W)

        # Decompose projection matrix into 'cam' intrinsics and rotation
        cam, R, t = cv2.decomposeProjectionMatrix(P2)[:3]

        # Store original intrinsics
        sample['orig_cam'] = cam.copy()

        # NOTE: We should multiply with world_to_cam, but difference is small
        sample['world_to_cam'] = np.eye(4)
        sample['world_to_cam'][:3, :3] = R
        sample['world_to_cam'][:3, 3] = -t[:3, 0]

        # Load depth map in meters
        depth_url = os.path.join(self.path, self.depths[index])

        # Break labels apart into separate entities (only if needed... slow!)
        if self.data_split != 'test' and self.train:
            labels = open(os.path.join(self.root, self.label_files[index])).readlines()
            sample['gt'] = []
            for label in [l[:-1].split(' ') for l in labels]:
                trunc = float(label[1])  # From 0 to 1 (truncated), how much object leaving image
                occ = int(label[2])  # 0 = visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
                alpha = float(label[3])  # Observation angle of object, ranging [-pi..pi]
                dimensions = [float(b) for b in label[8:11]]  # height, width, length (in meters)
                location = [float(b) for b in label[11:14]]  # 3D (ground) location in camera
                rot_y = float(label[14])  # Rotation around Y-axis in camera [-pi..pi]

                anno = {}
                anno['name'] = label[0]  # Describes the type of object: 'Car', 'Van', 'Truck'
                anno['bbox'] = [int(float(b)) for b in label[4:8]]  # LTRB in pixels
                anno['location'] = location
                anno['dimensions'] = dimensions
                anno['rotation_y'] = rot_y
                anno['alpha'] = alpha
                anno['score'] = 1
                anno['truncated'] = trunc
                anno['occluded'] = occ

                # Throw away all 3D information for unlabeled 3D boxes (set to -1000)
                anno['ignore'] = location[0] < -100

                sample['gt'].append(anno)

            # Some car instances are completely occluded by other things because annotated with LIDAR
            if True:
                for inst_i, anno_i in enumerate(sample['gt']):
                    for inst_j, anno_j in enumerate(sample['gt']):
                        if anno_i['name'] != 'Car' or inst_j == inst_i:
                            continue

                        # Compute Intersection normalized by anno_i's area
                        # Measures how much the box is subsumed by the other
                        inter_lt = np.maximum(anno_i['bbox'][:2], anno_j['bbox'][:2])
                        inter_br = np.minimum(anno_i['bbox'][2:], anno_j['bbox'][2:])
                        inter_wh = np.maximum(inter_br - inter_lt, 0)
                        intersection = (inter_wh[0] * inter_wh[1]) / ((anno_i['bbox'][2] - anno_i['bbox'][0]) *
                                                                      (anno_i['bbox'][3] - anno_i['bbox'][1]))

                        # Some 'DontCare's were simply put over other annotations... Jesus...
                        if intersection > 0.5 and anno_j['name'] == 'DontCare':
                            anno_i['ignore'] = True

                        # Check if 2D bbox fully inside another 2D bbox but Z larger, deactivate if true
                        if True:
                            if not anno_i['ignore'] and not anno_j['ignore']:
                                if anno_i['location'][2] > anno_j['location'][2] and intersection > 0.95:
                                    anno_i['ignore'] = True
                                    break

        # Filter all valid Car annotations based on type and difficulty
        annos = {'easy': [], 'medium': [], 'hard': []}
        for anno in sample['gt']:
            if anno['name'] != 'Car' or anno['ignore']:
                continue
            if is_anno_easy(anno):
                annos['easy'].append(anno)
            elif is_anno_moderate(anno):
                annos['medium'].append(anno)
            else:
                annos['hard'].append(anno)

        depth, pcd = get_kitti_frame(sample)
        sample['depth'] = depth
        sample['pcd'] = pcd
        sample['annos'] = annos

        return sample
