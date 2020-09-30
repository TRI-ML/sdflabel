import os
import torch
import torch.nn as nn
import torchvision.utils as vis

import utils.data as config
from networks.resnet_css import setup_css
from datasets.crops import Crops
from torch.utils.data import ConcatDataset


def train_css(cfgp):
    """
    Training of the CSS network

    Args:
        cfgp: Configuration parser
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set CSS
    css_path = config.read_cfg_string(cfgp, 'input', 'css_path', default='')
    css_net = setup_css(pretrained=True, model_path=css_path).to(device)

    # Set optimizer
    lr = config.read_cfg_float(cfgp, 'train', 'lr', default=1e-4)
    optimizer = torch.optim.Adam(css_net.parameters(), lr=lr)

    # Logs
    log_dir = config.read_cfg_string(cfgp, 'log', 'dir', default='log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Prepare the data
    batch_size = config.read_cfg_int(cfgp, 'train', 'batch_size', default=32)
    cpu_threads = config.read_cfg_int(cfgp, 'optimization', 'cpu_threads', default=3)
    data_path = config.read_cfg_string(cfgp, 'input', 'data_path', default=None)

    # Define training DB
    trainset = Crops(data_path)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=cpu_threads
    )

    # Losses
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()

    # Run the optimizer
    epochs = config.read_cfg_int(cfgp, 'train', 'epochs', default=1000)
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(trainloader):

            # Prepare input
            rgb = batch['rgb'].to(device)
            mask_gt = batch['mask'].to(device).long()
            uvw_gt = batch['uvw'].to(device)
            latent_gt = batch['latent'].squeeze(0).to(device)

            optimizer.zero_grad()

            # Get css output
            pred_css = css_net(rgb)
            u_pred, v_pred, w_pred = pred_css['u'], pred_css['v'], pred_css['w']
            uvw_pred_sm, mask_pred, mask_pred_sm = pred_css['uvw_sm'], pred_css['mask'], pred_css['mask_sm']
            lat_pred = pred_css['latent']

            # Define losses
            mask_gt_ext = mask_gt.unsqueeze(1).expand_as(u_pred).float()
            loss_u = criterion_ce(u_pred * mask_gt_ext, uvw_gt[:, 0] * mask_gt)  # U component
            loss_v = criterion_ce(v_pred * mask_gt_ext, uvw_gt[:, 1] * mask_gt)  # V component
            loss_w = criterion_ce(w_pred * mask_gt_ext, uvw_gt[:, 2] * mask_gt)  # W component
            loss_uvw = loss_u + loss_v + loss_w  # Full UVW cross-entropy loss
            loss_mask = criterion_ce(mask_pred, mask_gt) * 2  # Mask cross-entropy loss
            loss_latent = criterion_mse(lat_pred.squeeze(0), latent_gt)  # Latent vector MSE loss

            # Total loss
            loss = loss_uvw + loss_latent + loss_mask

            # Print current loss values
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLosses: global - {:.6f}, uvw - {:.6f}, mask - {:.6f}, latent - {:.6f}'
                .format(
                    epoch, batch_idx * len(rgb), len(trainloader.dataset), 100. * batch_idx / len(trainloader),
                    loss.item(), loss_uvw.item(), loss_mask.item(), loss_latent.item()
                )
            )
            loss.backward()
            optimizer.step()

        # Save net and visualize
        if (epoch + 1) % config.read_cfg_int(cfgp, 'log', 'analyse_epoch', default=10) == 0:

            # Store net
            net_dir = os.path.join(log_dir, 'net')
            if not os.path.exists(net_dir):
                os.makedirs(net_dir)
            torch.save(css_net.state_dict(), os.path.join(net_dir, 'css.pt'))

            # Visualize results
            if config.read_cfg_string(cfgp, 'log', 'plot', default=True):
                vis_dir = os.path.join(log_dir, 'vis')
                if not os.path.exists(vis_dir):
                    os.makedirs(vis_dir)

                uvw_masked_pred = uvw_pred_sm * mask_pred.argmax(dim=1, keepdim=True).expand_as(uvw_pred_sm).float()
                vis.save_image(
                    uvw_masked_pred.clone(), os.path.join(vis_dir, 'uvw_predsm_' + str(epoch) + '.png'), normalize=True
                )

                vis.save_image((uvw_gt.float() / 255.),
                               os.path.join(vis_dir, 'uvw_gt' + str(epoch) + '.png'),
                               normalize=True)
                vis.save_image(rgb.float(), os.path.join(vis_dir, 'uvw_gt_rgb' + str(epoch) + '.png'), normalize=True)
