import numpy as np
import math
import cv2
import open3d as o3d
import torch
from pyquaternion import Quaternion

precision = torch.float32
if precision == torch.float16:
    eps = 1e-4
else:
    eps = 1e-8


def is_anno_easy(anno):
    """
    Check if the difficulty of the KITTI annotation is "easy"
    Args:
        anno: KITTI annotation

    Returns: bool if "easy"

    """
    height = anno['bbox'][3] - anno['bbox'][1]
    if (anno['occluded'] > 0) or (anno['truncated'] > 0.15) or height < 40:
        return False
    return True


def is_anno_moderate(anno):
    """
    Check if the difficulty of the KITTI annotation is "moderate"
    Args:
        anno: KITTI annotation

    Returns: bool if "moderate"

    """
    height = anno['bbox'][3] - anno['bbox'][1]
    if (anno['occluded'] > 1) or (anno['truncated'] > 0.30) or height < 25:
        return False
    return True


def is_anno_hard(anno):
    """
    Check if the difficulty of the KITTI annotation is "hard"
    Args:
        anno: KITTI annotation

    Returns: bool if "hard"

    """
    height = anno['bbox'][3] - anno['bbox'][1]
    if (anno['occluded'] > 2) or (anno['truncated'] > 0.5) or height < 25:
        return False
    return True


def transform_bgr_crop(crop_bgr, orig=False):
    """
    Transform BGR image to Pytorch tensor
    Args:
        crop_bgr: input BGR image (OpenCV format)
        orig: output original image

    Returns: Pytorch tensor (and original image)

    """
    from PIL import Image
    import torchvision.transforms as transforms
    im_base = Image.fromarray(cv2.cvtColor((crop_bgr * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    im = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(im_base)
    if orig:
        im_orig = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])(im_base)
        return im, im_orig
    return im


def compute_depth_map(lidar, cam, w, h):
    """
    Compute a depth map given the LIDAR point cloud, camera matrix, and image dimensions
    Args:
        lidar: point cloud
        cam: camera matrix
        w: depth image width
        h: depth image height

    Returns: depth map

    """
    frustum = build_view_frustum(cam, 0, 0, w, h)
    inside_frustum = np.logical_and.reduce(frustum @ lidar.T > 0, axis=0)
    cam_xyz = lidar[inside_frustum]
    depth = np.zeros((h, w), dtype=np.float32)
    for (x, y), z in zip(project(cam, cam_xyz).astype(np.int32), cam_xyz[:, 2]):
        depth[y, x] = z
    return depth


def rot_from_yaw(yaw):
    """
    Get rotation matrix from yaw
    Args:
        yaw: float value representing yaw

    Returns: 3x3 rotation matrix

    """
    if not isinstance(yaw, torch.Tensor):
        yaw = torch.Tensor([yaw])
    cos = torch.cos(yaw)
    sin = torch.sin(yaw)
    z = yaw.new_tensor([0])
    o = yaw.new_tensor([1])

    rot = torch.stack((cos, z, sin, z, o, z, -sin, z, cos)).view(3, 3)
    return rot


def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def compute_iou(boxA, boxB):
    """
    Compute intersection over union
    Args:
        boxA: first bounding box
        boxB: second bounding box

    Returns: intersection over union

    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def roty_in_bev(pose):
    """
    Convert rotation about y axis (with convention of x-right, y-down, z-forward)
                    to BEV is x-z plane (with z forward)
    Args:
        pose (np.array): 3D pose of object

    Returns: rotation about y axis in BEV

    """
    # Rotate forward vector by pose
    forward_dir_rotated = (pose[:3, :3] @ np.asarray([0, 0, 1]).T).T

    # Measure angle from rotation origin
    rotation_y = math.acos(np.asarray([1, 0, 0]) @ (forward_dir_rotated))

    # We need to flip rotation_y when the rotated vector points forward
    if forward_dir_rotated[2] > 0:
        rotation_y *= -1
    return rotation_y


def alpha_in_bev(pose, rot_y):
    """    
    Args:
        pose (np.array): 3D pose of object

        rot_y float: Rotation about y (with convention of x-right, y-down, z-forward)        
                    i.e. BEV is x-z plane (with z forward)                                     

     """

    # Construct unit vector pointing in z direction (i.e. [0, 0, 1] direction)
    # NOTE: `car[0]` is end of unit vector, `car[1]` is start
    car = np.asarray([[0, 0, 1], [0, 0, 0]])

    # Tranform this unit vector by pose of car, and drop y component, thus keeping
    # heading direction in BEV (x-z grid)
    car = ((pose[:3, :3] @ car.T).T + pose[:3, 3])[:, ::2]

    # Getting positive theta angle (we define theta as the positive angle between
    # a ray from the origin through the base of the transformed unit vector and the z-axis
    theta = np.arctan2(abs(car[1, 0]), abs(car[1, 1]))

    # Depending on whether the base of the transformed unit vector is in the first or
    # second quadrant we add or subtract `theta` from `rot_y` to get alpha, respectively
    if car[1, 0] < 0:  # x of car location in BEV
        alpha = rot_y + theta
    else:
        alpha = rot_y - theta

    return alpha


def lookat(pos, target, up=np.asarray([0, 1, 0])):
    """Compute an OpenGL-style lookat matrix

    Args:
        pos (np.array): Position vector of camera
        target (np.array): Position vector of target 
        up (np.array): Defines the up direction of the camera  

    Returns:
        transform (np.array): 4x4 transformation matrix    
    """
    pos = np.asarray(pos)
    up = np.asarray(up)
    F = pos - np.asarray(target)
    f = F / np.linalg.norm(F)
    U = up / np.linalg.norm(up)
    s = np.cross(f, U)
    u = np.cross(s, f)
    M, T = np.eye(4), np.eye(4)
    M[:3, :3] = np.vstack([s, u, -f])
    T[:3, 3] = -pos
    transform = M @ T
    return transform


def build_correspondence_lineset(pts_A, pts_B, idxs):
    """Build a open3d.geometry.LineSet from a list of correspondences

    This function can both work with Numpy arrays and Torch tensors.

    Args:
        pts_A (np.array or torch.tensor): Point set in form (Nx3)
        pts_B (np.array or torch.tensor): Point set in form (Nx3)   
        idxs (list of int): marks correspondence between A[i] and B[idxs[i]]  

    Returns:
        line_set (open3d.geometry.LineSet)        
    """
    assert len(idxs) == len(pts_A)

    line_idx = [[i * 2 + 0, i * 2 + 1] for i in range(len(idxs))]

    if isinstance(pts_A, torch.Tensor):
        line_pts = torch.cat([pts_A, pts_B[idxs]], dim=1).view(-1, 3).detach().cpu().numpy()
    else:
        line_pts = np.reshape(np.concatenate([pts_A, pts_B[idxs]], axis=1), (-1, 3))

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_pts)
    line_set.lines = o3d.utility.Vector2iVector(line_idx)
    return line_set


def build_frustum_lineset(K, l, t, r, b):
    """Build a open3d.geometry.LineSet to represent a frustum

    Args:
        pts_A (np.array or torch.tensor): Point set in form (Nx3)
        pts_B (np.array or torch.tensor): Point set in form (Nx3)   
        idxs (list of int): marks correspondence between A[i] and B[idxs[i]]  

    Returns:
        line_set (open3d.geometry.LineSet)        
    """
    corners = np.asarray([(l, t), (r - 1, t), (r - 1, b - 1), (l, b - 1)], dtype=np.float32)

    rays = unproject(K, corners)
    rays /= np.linalg.norm(rays, axis=1)[:, None]

    line_idx = [[i * 2 + 0, i * 2 + 1] for i in range(4)]

    line_pts = []
    for ray in rays:
        line_pts.extend([[0, 0, 0], (ray * 100).tolist()])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_pts)
    colors = np.zeros((8, 3), dtype=np.uint8)
    colors[:, 1] = 255
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.lines = o3d.utility.Vector2iVector(line_idx)
    return line_set


def build_vizbox(bbox3d, rgb=[1, 0, 0]):
    """Build a open3d.geometry.LineSet to represent a cuboid

    Args:
        bbox3d (np.array): Point set in form (Nx3), following canonical format
        rgb (list of float): color of cuboid

    Returns:
        line_set (open3d.geometry.LineSet)        
    """
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(bbox3d)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    colors = [rgb for i in range(len(lines))]
    # Paint upper front bar in opposite color
    colors[0] = [1 - rgb[0], 1 - rgb[1], 1 - rgb[2]]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def reproject(color, depth, K, flip_color_channels=False, filter=False):
    """Reprojects a depth map to sparse list of 3D colored points.

    This function can both work with Numpy arrays and Torch tensors.

    Args:
        color (np.array or torch.tensor): Color image either (3xHxW) or (HxWx3)
        depth (np.array or torch.tensor): Depth image either (1xHxW) or (HxW)       
        K (np.array or torch.tensor): Camera intrinsics (3x3)         
        flip_color_channels (bool): If true, colors will be flipped (BGR2RGB)

    Returns:
        points (np.array or torch.tensor): 3D points (Nx3)
        colors (np.array or torch.tensor): Points colors (Nx3)

    """
    if isinstance(depth, torch.Tensor):
        depth_ = depth.squeeze()
        y, x = torch.nonzero(depth_).split(1, dim=1)
        good_xy = torch.cat((x, y), dim=1).float()
        homo_points = torch.cat([good_xy, good_xy.new_ones(len(good_xy), 1)], dim=1)
        point3D = (torch.inverse(K).to(homo_points.device) @ homo_points.t()).t()
        points = point3D * depth_[y, x]
        # Check for the special (render) case where the channels go first
        if color.shape[0] == 3:
            colors = color[:, y, x].squeeze().t()
        else:
            colors = color[y, x].squeeze()
        if flip_color_channels:
            colors = torch.stack((colors[:, 2], colors[:, 1], colors[:, 0]), axis=1)

    else:
        y, x = np.nonzero(depth)
        good_xy = np.stack((x, y), axis=1).astype(np.float32)
        homo_points = np.concatenate((good_xy, np.ones((len(good_xy), 1))), axis=1)
        point3D = (np.linalg.inv(K) @ homo_points.T).T
        points = point3D * depth[y, x][:, None]
        colors = color[y, x]
        if flip_color_channels:
            colors = colors[:, ::-1]

    if len(colors.shape) < 2:
        colors = colors.unsqueeze(0)

    if filter:
        # Select non-black (foreground) NOCS points
        active = (colors > 0).sum(dim=1) > 0
        points = points[active]
        colors = colors[active]

    return points, colors


def build_heatmap(input, min=None, max=None):
    """ Returns a RGB heatmap representation """
    if min is None:
        min = np.amin(input)
    if max is None:
        max = np.amax(input)
    rescaled = 255 * ((input - min) / (max - min))
    final = cv2.applyColorMap(rescaled.astype(np.uint8), cv2.COLORMAP_JET)
    return final.astype(np.float32) / 255


def transform_label(bbox3d):
    """ Takes (8, 3) array of 3D bounding box points and returns
    centroid, direction vector, bounding box extend from the center of object
    """

    # center of mass
    centroid = np.mean(bbox3d, axis=0)

    # direction vector (centroid of 4 3D front points - object centroid)
    direction = np.mean(bbox3d[:4], axis=0) - centroid
    direction /= np.linalg.norm(direction)  # Make unit vector

    # Extend of the BB (in local object system)
    width = np.linalg.norm(bbox3d[0] - bbox3d[1])  # left-right difference
    height = np.linalg.norm(bbox3d[0] - bbox3d[3])  # up-down difference
    length = np.linalg.norm(bbox3d[0] - bbox3d[4])  # front-back difference

    return centroid, direction, np.asarray([width, height, length])


def transform_kitti_to_cuboid(width, height, length, location, rot_y):
    """
    Generate a cuboid from KITTI label parameters
    Args:
        width: width of the bounding box
        height: height of the bounding box
        length: length of the bounding box
        location: location of the bottom of the (Y=0) bounding box
        rot_y: Rotation along Y
    """
    # Build box from KITTI3D label at origin. The car sits on the ground at Y=0
    # NOTE: Invert Y-axis because OpenCV goes down
    w, h, l = width, height, length
    front = np.asarray([[-w / 2, -h, l / 2], [+w / 2, -h, l / 2], [+w / 2, +0, l / 2], [-w / 2, +0, l / 2]])
    back = np.copy(front)
    back[:, 2] *= -1
    local_box = np.vstack((front, back))

    # Rotate and translate into global space
    # 'A car which is facing along the X-axis of the camera coordinate system
    # corresponds to rotation_y=0'. This is why we add PI/2 here...
    angle = rot_y + np.pi / 2
    rot = Quaternion(axis=[0, 1, 0], radians=angle).rotation_matrix
    return (rot @ local_box.T).T + location


def project(K, p3d):
    p2d = cv2.projectPoints(p3d, (0, 0, 0), (0, 0, 0), K, None)[0]
    return p2d.squeeze().astype(np.float32)


def unproject(K, p2d):
    rays = cv2.undistortPoints(p2d[:, None], K, None)
    return cv2.convertPointsToHomogeneous(rays).reshape(-1, 3).astype(np.float32)


def build_view_frustum(K, l, t, r, b):
    # https://gamedev.stackexchange.com/questions/79172/checking-if-a-vector-is-contained-inside-a-viewing-frustum
    # Shoot rays through each image corner
    corners = np.asarray([(l, t), (r - 1, t), (r - 1, b - 1), (l, b - 1)], dtype=np.float32)

    rays = unproject(K, corners)
    rays /= np.linalg.norm(rays, axis=1)[:, None]

    # Build the 4 plane normals that all point towards into frustum interior
    top = np.cross(rays[0], rays[1])
    right = np.cross(rays[1], rays[2])
    bottom = np.cross(rays[2], rays[3])
    left = np.cross(rays[3], rays[0])
    frustum = np.stack((top, right, bottom, left))
    return frustum


def build_cam_frustum(K, img_w, img_h):
    return build_view_frustum(K, 0, 0, img_w, img_h)


def get_kitti_label(dsdf, grid, latent, scale, trans, yaw, p_WC, bbox):
    """
    Reconstruct KITTI label from the estimated parameters
    Args:
        dsdf: DeepSDF network
        grid: Grid instance
        latent: Estimated latent vector for DeepSDF
        scale: Estimated scale
        trans: Estimated translation vector
        yaw: Estimated rotation
        p_WC: Lidar -> Camera matrix
        bbox: 2D bounding box

    Returns:

    """
    # Define device and precision
    precision = grid.points.dtype
    device = grid.points.device

    results = {'yaw': yaw.detach().cpu().numpy(), 'trans': trans.detach().cpu().numpy()}
    results['scale'] = scale.detach().cpu().numpy()
    results['latent'] = latent.detach().cpu().numpy()

    # Build final transformation from latent space -> camera frame -> world frame
    cam_T = np.eye(4)
    cam_T[:3, :3] = rot_from_yaw(results['yaw'].item()).detach().cpu().numpy() @ np.diag([1, -1, 1])
    cam_T[:3, 3] = results['trans'] * results['scale']

    # The final pose is in camera space, since we transformed lidar into camera.
    # Need a final conversion into global space for proper evaluation.
    global_T = np.linalg.inv(p_WC) @ cam_T

    ### Compute all entities needed for KITTI3D eval
    # Scale the SDF in its local frame and compute the extent
    inputs = torch.cat([latent.expand(grid.points.size(0), -1), grid.points], 1).to(latent.device, latent.dtype)
    pred_sdf_grid, inv_scale = dsdf(inputs)
    points_masked, _, _ = grid.get_surface_points(pred_sdf_grid)

    # points_masked, normals_masked, _ = dsdf_tools.run_dsdf(dsdf, latent.to(precision), point_inputs)
    scaled_points = points_masked.detach().cpu().numpy() * results['scale'][None]
    xmin, xmax = scaled_points[:, 0].min(), scaled_points[:, 0].max()
    ymin, ymax = scaled_points[:, 1].min(), scaled_points[:, 1].max()
    zmin, zmax = scaled_points[:, 2].min(), scaled_points[:, 2].max()
    width, height, length = xmax - xmin, ymax - ymin, zmax - zmin
    bottom_center = np.asarray([0, ymin, 0])
    render_pose = torch.eye(4).to(device, precision)
    render_pose[:3, :3] = rot_from_yaw(yaw).to(device, precision)
    render_pose[1] *= -1  # Flip Y for rendering
    render_pose[:3, 3] = trans.to(precision)

    # Generate a label
    label = {'name': 'Car'}
    label['bbox'] = bbox  # Take original 2D bbox since their projection might differ
    #label['bottom_center'] = bottom_center
    label['location'] = (global_T[:3, :3] @ bottom_center.T).T + global_T[:3, 3]
    label['dimensions'] = [height, width, length]
    label['rotation_y'] = roty_in_bev(global_T)
    label['alpha'] = alpha_in_bev(global_T, label['rotation_y'])
    label['score'] = 1

    return label, scaled_points, cam_T


def get_annos(diff_annos, sample):
    """
    Get annotations based on their difficulty
    
    Args:
        diff_annos: Annotation's difficulty 
        sample: Sample object

    Returns:

    """
    if diff_annos == 'hard':
        annos = sample['annos']['easy'] + sample['annos']['medium'] + sample['annos']['hard']
    elif diff_annos == 'medium':
        annos = sample['annos']['easy'] + sample['annos']['medium']
    else:
        annos = sample['annos']['easy']
    annos = sorted(annos, key=lambda i: i['location'][2])  # Sort annos based on ascending depth
    return annos


def adjust_intrinsics_crop(K, crop_size, bbox, max_crop_area):
    """
    Adjust intrinsics to render the crop 
    
    Args:
        sample: intrinsics parameters
        crop_size: size of the crop
        bbox: 2D bounding box
        max_crop_area: Maximum crop area (to assure 

    Returns: new crop size, adjusted intrinsics, and original intrinsics

    """
    l, t, r, b = bbox
    crop_H, crop_W = crop_size
    area_ratio_sqrt = math.sqrt(max_crop_area / (crop_H * crop_W))
    crop_size = (crop_size * area_ratio_sqrt)
    crop_size = crop_size.int().numpy().tolist()
    intrinsics = torch.Tensor(K)
    intrinsics[0, 2] -= l
    intrinsics[1, 2] -= t
    off_intrinsics = intrinsics.clone()
    intrinsics[:2] *= area_ratio_sqrt
    return crop_size, intrinsics, off_intrinsics


def get_kitti_frame(sample):
    """
    Get KITTI depth image and point cloud
    Args:
        sample: KITTI sample object

    Returns: scene depth and point cloud

    """
    H, W, _ = sample['image'].shape

    # Filter out lidar points outside field of view
    scene_lidar = sample['lidar']
    frustum = build_view_frustum(sample['orig_cam'], 0, 0, W, H)
    scene_lidar = scene_lidar[np.logical_and.reduce(frustum @ scene_lidar.T > 0, axis=0)]

    # Build Open3D pcd and estimate normals
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_lidar)
    scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    # Estimate road plane (stupidly by checking orthogonality to camera. RANSAC better)
    normals = np.asarray(scene_pcd.normals)
    ortho_to_cam = np.abs(normals @ np.asarray([0, 1, 0])) > 0.9
    plane_points = scene_lidar[ortho_to_cam]
    plane_normal = np.mean(normals[ortho_to_cam], axis=0)
    plane_normal /= np.linalg.norm(plane_normal)
    plane_dists = plane_normal @ plane_points.T
    plane_offset = np.median(plane_dists)

    # Filter out road plane by simple normal check
    if True:
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
