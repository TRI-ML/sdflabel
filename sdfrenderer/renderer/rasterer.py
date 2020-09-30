import numpy as np
import torch

from renderer.utils_rasterer import calibration_matrix
from renderer.projection import project_in_2D, project_in_2D_quat
from renderer.primitives import inside_circle, inside_circle_opt, inside_surfel


class Rasterer(torch.nn.Module):
    def __init__(self, K, resolution_px, diagonal_mm=20, focal_len_mm=70, precision=torch.float32):
        """
        Rasterizer constructor

        Args:
            K (torch.Tensor): Intrinsic camera parameters (3,3)
            resolution_px (tuple): Camera resolution in pixels
            diagonal_mm: Camera focal length in millimeters
            focal_len_mm: Limit the maximum amount of point, to avoid memory run-outs
            precision: Precision of the used variables
        """
        super(Rasterer, self).__init__()
        self.res_x_px, self.res_y_px = resolution_px  # image resolution in pixels

        # Prepare the 2d grid once
        yy, xx = np.mgrid[0:self.res_y_px, 0:self.res_x_px]
        grid = np.concatenate((xx[..., None], yy[..., None]), axis=-1)
        self.register_buffer('grid', torch.from_numpy(grid.reshape((1, -1, 2))))

        # Prepare grid for primitives
        yy, xx = np.mgrid[-7:8, -7:8]
        grid_prim = np.concatenate((xx[..., None], yy[..., None]), axis=-1)
        self.register_buffer('grid_prim', torch.from_numpy(grid_prim.reshape((1, -1, 2))))

        # Store the calibration matrix
        if K is None:
            K = calibration_matrix(
                resolution_px=(self.res_x_px, self.res_y_px),
                diagonal_mm=diagonal_mm,
                focal_len_mm=focal_len_mm,
                skew=0
            )
            self.register_buffer('K', torch.from_numpy(K).to(precision))
        else:
            self.register_buffer('K', K.to(precision))

    def __call__(self, *args, **kwargs):
        return super(Rasterer, self).__call__(*args, **kwargs)

    def forward(
        self,
        coords,
        normals,
        colors,
        camera_matrix,
        rot='quat',
        primitives='disc',
        bg=None,
        output_mask=False,
        output_depth=False,
        output_normals=False,
        output_nocs=False,
        output_points=True
    ):

        # Project current points to 2D
        if rot == 'dcm':
            points_proj = project_in_2D(
                self.K,
                camera_matrix,
                coords,
                normals,
                colors,
                resolution_px=(self.res_x_px, self.res_y_px),
                output_nocs=output_nocs
            )
        elif rot == 'quat':
            points_proj = project_in_2D_quat(
                self.K,
                camera_matrix,
                coords,
                normals,
                colors,
                resolution_px=(self.res_x_px, self.res_y_px),
                output_nocs=output_nocs
            )

        vertices_3d = points_proj['points_3d']
        vertices_2d = points_proj['points_2d']
        normals = points_proj['normals_3d']
        colors = points_proj['colors_3d']

        # Select primitives
        if primitives == 'circle':
            prob_color = inside_circle(
                self.K, self.grid, vertices_2d, vertices_3d, normals, diam=0.02, add_bg=(bg is not None)
            )
        elif primitives == 'circle_opt':
            prob_color = inside_circle_opt(
                self.K, self.grid_prim, vertices_2d, vertices_3d, normals, diam=0.025, add_bg=(bg is not None)
            )
        elif primitives == 'disc':
            prob_color = inside_surfel(
                self.K, self.grid, vertices_2d, vertices_3d, normals, diam=0.04, softclamp=False, add_bg=(bg is not None)
            )

        # Add background if available
        if bg is not None:
            normals_ext = ((normals + 1) / 2).unsqueeze(-1).expand_as(prob_color[:-1, :, :])  # object normals + bg
            colors_ext = ((colors + 1) / 2).unsqueeze(-1).expand_as(prob_color[:-1, :, :])  # object colors + bg
            colors_ext = torch.cat([colors_ext, bg.view(1, colors_ext.size(1),
                                                        colors_ext.size(2))])  # background colors
        else:
            if output_nocs:
                colors_ext = ((colors + 1) / 2).unsqueeze(-1).expand_as(prob_color)  # object colors
            else:
                colors_ext = (colors).unsqueeze(-1).expand_as(prob_color)  # object colors
            normals_ext = ((normals + 1) / 2).unsqueeze(-1).expand_as(prob_color)  # object normals + bg

        # Compose an image
        rendering = {}

        # Color / NOCS
        rendering_color = prob_color * colors_ext
        rendering['color'] = torch.clamp(torch.sum(rendering_color, dim=0).view(3, self.res_y_px, self.res_x_px), max=1)

        # Mask
        if output_mask:
            prob_mask_depth = prob_color[:, :1, :]
            rendering['mask'] = torch.clamp(
                torch.sum(prob_mask_depth, dim=0).view(1, self.res_y_px, self.res_x_px), max=1
            )

        # Depth
        if output_depth:
            prob_mask_depth = prob_color[:, :1, :]
            rendering_depth = prob_mask_depth * vertices_3d[:, 2:].unsqueeze(-1).expand_as(prob_mask_depth)
            rendering['depth'] = torch.sum(rendering_depth, dim=0).view(1, self.res_y_px, self.res_x_px)

        # Normals
        if output_normals:
            rendering_normals = prob_color * normals_ext
            rendering['normals'] = torch.clamp(
                torch.sum(rendering_normals, dim=0).view(3, self.res_y_px, self.res_x_px), max=1
            )

        # Points
        if output_points:
            points = {}
            points['xyz'] = vertices_3d
            points['rgb'] = (colors + 1) / 2
            points['xyzf'] = points_proj['points_3d_filt']
            points['rgbf'] = (points_proj['colors_3d_filt'] + 1) / 2
            return rendering, points

        return rendering
