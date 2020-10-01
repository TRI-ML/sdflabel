import os
import torch
import trimesh
import argparse
import torchvision.utils as vis
from scipy.spatial.transform import Rotation as R

from renderer.rasterer import Rasterer
from grid import Grid3D
import deepsdf.workspace as dsdf_ws


def render_model(model_path, primitives, precision, output_dir='renderer/output'):
    """
    Render a colored point cloud

    Args:
        model_path: Path to the model file
        primitives: Type of rendering primitives, i.e. circle, circle_opt, disc
        precision: Precision, i.e. float16 or float32
    """
    device = 'cuda'
    if device not in ['cpu', 'cuda']:
        raise ValueError('Unknown device.')

    # Camera pose
    rot = 90
    r = R.from_euler('x', rot, degrees=True)  # rotation
    trans = torch.Tensor([0., 0., 10]).to(device, precision)  # translation
    camera_pose_dcm = torch.eye(4).to(device, precision)
    camera_pose_dcm[:3, :3] = torch.Tensor(r.as_dcm())
    camera_pose_dcm[:3, 3] = trans

    # Load the model
    model_vertices = torch.Tensor(trimesh.load(model_path).vertices / trimesh.load(model_path).vertices.max()).to(device, precision)
    model_normals = torch.Tensor(trimesh.load(model_path).vertex_normals).to(device, precision)
    model_colors = torch.Tensor(trimesh.load(model_path).visual.vertex_colors[:, :3]).to(device, precision) / 255

    # Create a renderer instance
    crop_size = (200, 100)
    renderer = Rasterer(None, crop_size, precision=precision).to(device)

    # Render the model
    rendering = renderer(
        model_vertices,
        model_normals,
        model_colors,
        camera_pose_dcm.to(device),
        rot='dcm',
        bg=None,
        output_normals=True,
        primitives=primitives
    )
    transformed_pc = rendering[1]
    rendered_image = rendering[0]

    # Save image
    os.makedirs(output_dir, exist_ok=True)
    vis.save_image(rendered_image['color'], os.path.join(output_dir, 'demo_cad.png'))


def render_sdf(path_dsdf, primitives, precision, output_dir='renderer/output'):
    """
    Render the output of the DeepSDF net

    Args:
        path_dsdf: Path to the DeepSDF net's weights
        primitives: Type of rendering primitives, i.e. circle, circle_opt, disc
        precision: Precision, i.e. float16 or float32
    """
    device = 'cuda'
    if device not in ['cpu', 'cuda']:
        raise ValueError('Unknown device.')

    # Camera pose
    rot = 90
    r = R.from_euler('y', rot, degrees=True)  # rotation
    trans = torch.Tensor([0., 0., 10]).to(device, precision)  # translation
    camera_pose_dcm = torch.eye(4).to(device, precision)
    camera_pose_dcm[:3, :3] = torch.Tensor(r.as_dcm())
    camera_pose_dcm[:3, 3] = trans

    # Load DSDF
    dsdf, latent_size = dsdf_ws.setup_dsdf(path_dsdf, precision=precision)
    dsdf = dsdf.to(device)

    # Create 3D grid and form DeepSDF input
    grid_density = 40  # dummy grid density
    grid_3d = Grid3D(grid_density, device, precision)
    latent = torch.Tensor([1, 0, 0]).to(device, precision)  # dummy latent code
    inputs = torch.cat([latent.expand(grid_3d.points.size(0), -1), grid_3d.points],
                                          1).to(latent.device, latent.dtype)

    # Get DeepSDF output
    pred_sdf_grid, inv_scale = dsdf(inputs)

    # Get surface points using 0-isosurface projection
    pcd_dsdf, nocs_dsdf, normals_dsdf = grid_3d.get_surface_points(pred_sdf_grid)

    # Create a renderer instance
    crop_size = (200, 100)
    renderer = Rasterer(None, crop_size, precision=precision).to(device)

    # Render SDF
    rendering = renderer(
        pcd_dsdf,
        normals_dsdf,
        normals_dsdf,
        camera_pose_dcm.to(device),
        rot='dcm',
        bg=None,
        output_normals=True,
        primitives=primitives,
        output_nocs=True
    )
    transformed_pc = rendering[1]
    rendered_image = rendering[0]

    # Save image
    os.makedirs(output_dir, exist_ok=True)
    vis.save_image(rendered_image['color'], os.path.join(output_dir, 'demo_dsdf.png'))


def main():
    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='/models/pd_kitti', help='path to the model - 3D (ply, obj) or DeepSDF network')
    parser.add_argument('--primitives', '-p', default='disc', help='circle, circle_opt, disc')
    parser.add_argument('--precision', '-type', default=torch.float16, help='float16, float32')

    # Parse arguments
    args = parser.parse_args()

    # Execution
    if os.path.splitext(args.model)[1] == '.pt':
        # - render SDF
        render_sdf(args.model, args.primitives, args.precision)
    else:
        # - render point cloud
        render_model(args.model, args.primitives, args.precision)


if __name__ == '__main__':
    main()
