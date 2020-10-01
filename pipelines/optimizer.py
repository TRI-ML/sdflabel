import open3d as o3d
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from sklearn.neighbors import KDTree

from renderer.rasterer import Rasterer
import utils.refinement as rtools
import utils.visualizer as viztools


class MultipleOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


def get_opt_params(params, device):

    # Transform to torch
    for key, value in params.items():
        params[key] = Variable(torch.Tensor(value).to(device, torch.float32), requires_grad=True)

    # Generate optimization parameter list
    optim_params = []
    optim_params += [{'params': params['yaw'], 'lr': 0.01}]
    # optim_params += [{'params': params['quat'], 'lr': 0.005}]
    optim_params += [{'params': params['trans'], 'lr': 0.01}]
    optim_params += [{'params': params['scale'], 'lr': 0.01}]
    optim_params += [{'params': params['latent'], 'lr': 0.00003}]

    return params, optim_params


class Optimizer:
    def __init__(self, params, device, weights, rot='dcm'):

        self.params, self.optim_params = get_opt_params(params, device)
        self.optim_params_adam = self.optim_params[:2]
        self.optim_params_sgd = self.optim_params[2:]
        self.solver = MultipleOptimizer(
            torch.optim.Adam(self.optim_params_adam, lr=0.03),
            torch.optim.SGD(self.optim_params_sgd, lr=0.01, momentum=0.0)
        )
        self.weights = weights
        self.rot = rot

    def optimize(self, iters_optim, nocs_pred, pcd_frustum_np, dsdf, grid, K, crop_size, viz_type=None, frame_vis=None):
        """
        Optimization loop
        Args:
            iters_optim (int): Number of iterations
            nocs_pred (torch.Tensor): CSS network prediction
            pcd_frustum_np (np.array): LIDAR point cloud (N,3)
            dsdf: DeepSDF network
            grid: Grid object
            K (torch.Tensor): Camera matrix (3,3)
            crop_size (list): Size of the optimized crop
            viz_type ('2d'/'3d'): Type of the visualization
            frame_vis: Image of the full frame
        """
        self.device = grid.points.device
        self.precision = grid.points.dtype
        self.renderer = Rasterer(K, crop_size[::-1], precision=K.dtype).to(K.device)

        # Create an open3D visualizer instance if needed
        if viz_type == '3d':
            viz = o3d.visualization.Visualizer()
            viz.create_window(width=800, height=600)

        for e in range(iters_optim):

            self.solver.zero_grad()

            # Apply inverse scale to scene to not screw up renderer
            pcd_frustum = (torch.Tensor(pcd_frustum_np).to(self.device) / self.params['scale']).to(self.precision)

            # Build pose
            render_pose = torch.eye(4).to(self.device, self.precision)
            render_pose[:3, :3] = rtools.rot_from_yaw(self.params['yaw']).to(self.device, self.precision)
            render_pose[1] *= -1  # Flip Y for rendering
            render_pose[:3, 3] = self.params['trans'].to(self.precision)

            # quat_ = F.normalize(self.params['quat'], p=2, dim=0).to(self.precision)
            # render_pose = torch.cat([quat_, self.params['trans'].to(self.precision)])

            # Normalize latent space into sphere.
            latent_ = F.normalize(self.params['latent'].to(self.precision), p=2, dim=0)

            #  DeepSDF Inference
            inputs = torch.cat([latent_.expand(grid.points.size(0), -1), grid.points],
                               1).to(latent_.device, latent_.dtype)
            pred_sdf_grid, inv_scale = dsdf(inputs)

            # Surface point/normal extraction
            pcd_dsdf, _, normals_dsdf = grid.get_surface_points(pred_sdf_grid)

            # Zero the gradients from the surface point estimation
            self.solver.zero_grad()

            # Render surface points and normals and backproject
            rendering, points = self.renderer(
                pcd_dsdf,
                normals_dsdf,
                normals_dsdf,
                render_pose,
                primitives='disc',
                rot=self.rot,
                bg=None,
                output_depth=False,
                output_normals=True,
                output_nocs=True,
                output_points=True,
                output_mask=True
            )
            pcd_dsdf_trans, clrs_dsdf = points['xyzf'], points['rgbf']

            # Compute nearest neighbors and threshold based on metric distance
            if (pcd_dsdf_trans.nelement() == 0) or (pcd_frustum.nelement() == 0):
                print('Skip frame')
                continue

            # 3D Loss
            loss_3d, dists_3d, idxs_3d = self.compute_loss_3d(pcd_dsdf_trans, pcd_frustum)

            # Projective NOCS loss (masked)
            target_2d = F.interpolate(
                nocs_pred.unsqueeze(0), size=rendering['color'].shape[1:], mode='nearest'
            ).squeeze(0)
            #target_2d = torch.load('target.pt').float()

            # 2D Loss
            #loss_2d = torch.norm(rendering['color'] - target_2d.to(self.precision), dim=0).mean()  # L2 between points
            loss_2d = self.compute_loss_2d(rendering['color'], target_2d)

            weight_2d = self.weights['2d']
            weight_3d = self.weights['3d']
            loss = weight_3d * loss_3d + weight_2d * loss_2d

            # Check for nans
            if torch.isnan(loss).sum() > 0 or loss.sum() == 0:
                print('Skip frame')
                continue

            # Plot losses
            print('ITER {} | Losses: 2D - {}, 3D - {}, Total - {}'.format(e, weight_2d * loss_2d.item(),
                                                                          weight_3d * loss_3d.item(), loss.item()))
            loss.backward()
            self.solver.step()

            # Visualize
            if viz_type == '2d' or viz_type == '3d':
                viztools.plot_full_frame(frame_vis, rendering['normals'])
                viztools.plot_patches(rendering['color'], target_2d)
            if viz_type == '3d':
                viztools.plot_3d(viz, pcd_frustum, pcd_frustum, pcd_dsdf_trans, clrs_dsdf, dists_3d, idxs_3d, interactive=False)

    def compute_loss_3d(self, pcd_dsdf_trans, pcd_frustum, threshold=0.2):
        """
        Compute 3D loss between the estimated and LIDAR point clouds computed as a mean of point pair distances between 2 point clouds
        Args:
            pcd_dsdf_trans (torch.Tensor): Estimated point cloud (N,3)
            pcd_frustum (torch.Tensor): LIDAR point cloud (N,3)
            threshold (float): maximum allowed distance between the point pairs to be considered for the loss

        Returns: 3D loss value

        """
        if (pcd_dsdf_trans.nelement() != 0) and (pcd_frustum.nelement() != 0):

            # Estimate nearest neighbors (distances and ids) between given point clouds
            kdtree = KDTree(pcd_frustum.detach().cpu().numpy())
            dists, idxs = kdtree.query(pcd_dsdf_trans.detach().cpu().numpy())

            # Reformat distances and idxs to 1-dim array
            idxs = np.asarray([val for sublist in idxs for val in sublist])
            dists = np.asarray([val for sublist in dists for val in sublist])

            # Measure distances between filtered point pairs
            close_by = dists < threshold / self.params['scale'][0].item()
            dists_thres = (pcd_frustum[idxs[close_by]] - pcd_dsdf_trans[close_by]).norm(p=2, dim=1)

            # Check if point pairs exist
            if (dists_thres.nelement() != 0):
                loss_3d = dists_thres.mean()
            else:
                loss_3d = torch.tensor(0).to(self.device, self.precision)
        else:
            loss_3d = torch.tensor(0).to(self.device, self.precision)
        return loss_3d, dists, idxs

    def compute_loss_2d(self, rendering_nocs, css_nocs, diam=5, threshold_nocs=1):
        """
        Compute 2D loss between the CSS net output and the rendering
        Args:
            rendering_nocs (torch.Tensor): NOCS image from the renderer
            css_nocs (torch.Tensor): NOCS image prediction from the CSS network
            diam (int): pixel distance defining the CSS NOCS masks diameter
            threshold_nocs (float): maximum allowed NOCS distance to be considered for the loss

        Returns: 2D loss value

        """
        # Get idxs of nonzero elements
        rendering_nonzero_idxs = rendering_nocs.sum(0).nonzero()
        if rendering_nonzero_idxs.sum():

            # Construct a 2D meshgrid dimensionally equal to the rendering
            xx, yy = torch.meshgrid(torch.arange(rendering_nocs.shape[1]),
                                    torch.arange(rendering_nocs.shape[2]))
            grid_2d = torch.cat((xx.unsqueeze(-1), yy.unsqueeze(-1)), -1).float().to(self.device, self.precision)
            grid_2d_ext = grid_2d.unsqueeze(0).expand(rendering_nonzero_idxs.shape[0], *grid_2d.size())
            grid_2d_ext = grid_2d_ext.reshape(rendering_nonzero_idxs.shape[0], -1, 2)

            # For each non-zero rendering point define a circular area on css_nocs based on a given diameter
            vectors_to_point = grid_2d_ext - rendering_nonzero_idxs.view(-1, 1, 2)
            dist_to_point = torch.clamp(diam - vectors_to_point.pow(2).sum(-1).sqrt(), min=0)
            dist_to_point = dist_to_point.reshape(dist_to_point.shape[0], *grid_2d[:, :, 0].size())
            css_nocs_masked = css_nocs.unsqueeze(0) * dist_to_point.unsqueeze(1)

            # Compute minimum distances between rendering points and css_nocs masked points
            rendering_nonzeros = rendering_nocs[:, rendering_nocs.sum(0).nonzero(as_tuple=True)[0],
                             rendering_nocs.sum(0).nonzero(as_tuple=True)[1]].transpose(1, 0)
            diff = (css_nocs_masked - rendering_nonzeros.unsqueeze(-1).unsqueeze(-1)).pow(2).sum(1).sqrt()
            diff_min = diff.view(diff.shape[0], -1).min(1)[0]
            loss_2d = diff_min[diff_min < threshold_nocs].mean()
        else:
            loss_2d = torch.tensor(0).to(self.device, self.precision)
        return loss_2d
