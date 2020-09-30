import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as transforms
import random


class Crops(Dataset):
    def __init__(self, path):

        # Parse the GT file
        self.path = path
        with open(os.path.join(path, 'crops.json'), 'r') as f:
            self.gt = json.load(f)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):

        # Get sample data
        gt_sample = self.gt[str(idx)][0]

        # Read image and depth map
        rgb_orig = Image.open(os.path.join(self.path, '{:05d}'.format(idx) + '_rgb.png')).convert('RGB')
        uvw_orig = Image.open(os.path.join(self.path, '{:05d}'.format(idx) + '_uvw.png')).convert('RGB')
        crop_size = torch.Tensor(rgb_orig.size)

        # Get latent vector
        latent = np.array(gt_sample['latent'])

        # Get pose
        extrinsics = np.array(gt_sample['extrinsics']).reshape((4, 4))
        quat = R.from_dcm(extrinsics[:3, :3]).as_quat()
        quat = np.concatenate([quat[3:], quat[:3]])  # reformat to (w, x, y, z)
        z = extrinsics[2, 3] / 100

        # Get camera parameters
        intrinsics = np.array(gt_sample['intrinsics']).reshape((3, 3))

        # Random transformations
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformation_rgb = transforms.Compose([
            transforms.RandomRotation(10, Image.BILINEAR, expand=True),
            transforms.Resize((128, 128)),
            transforms.RandomResizedCrop(128, scale=(0.5, 1.0)),
            transforms.ToTensor(),
        ])
        transformation_uvw = transforms.Compose([
            transforms.RandomRotation(10, Image.NEAREST, expand=True),
            transforms.Resize((128, 128), Image.NEAREST),
            transforms.RandomResizedCrop(128, scale=(0.5, 1.0), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

        # Color jitter
        color_aug = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
        rgb = color_aug(rgb_orig)

        # Normalization
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # keep same seed for transformations
        rgb = normalize(transformation_rgb(rgb))
        random.seed(seed)  # keep same seed for transformations
        uvw = (transformation_uvw(uvw_orig) * 255).long()
        random.seed(seed)  # keep same seed for transformations
        mask = (uvw.sum(0) > 0).long()  # mask

        # Transform to torch format
        latent = torch.from_numpy(latent)

        # Store sample dict
        sample = dict()
        sample['rgb'] = rgb.float()
        sample['uvw'] = uvw.long()
        sample['mask'] = mask.long()
        sample['latent'] = latent.float()
        sample['crop_size'] = crop_size.long()
        sample['intrinsics'] = torch.Tensor(intrinsics).float()
        sample['pose'] = torch.Tensor(extrinsics).float()

        return sample
