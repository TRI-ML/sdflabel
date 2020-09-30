import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from networks.unet_parts import up, outconv
import torch
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def project_vecs_onto_sphere(vectors, radius, surface_only=True):
    for i in range(len(vectors)):
        v = vectors[i]
        length = torch.norm(v).detach()

        if surface_only or length.cpu().data.numpy() > radius:
            vectors[i] = vectors[i].mul(radius / (length + 1e-8))
    return vectors


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def _freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # head 1 - u
        self.up1_u = up(384, 128)
        self.up2_u = up(192, 64)
        self.up3_u = up(128, 64)
        self.up4_u = up(64, 64, add_shortcut=False)

        # head 2 - v
        self.up1_v = up(384, 128)
        self.up2_v = up(192, 64)
        self.up3_v = up(128, 64)
        self.up4_v = up(64, 64, add_shortcut=False)

        # head 3 - w
        self.up1_w = up(384, 128)
        self.up2_w = up(192, 64)
        self.up3_w = up(128, 64)
        self.up4_w = up(64, 64, add_shortcut=False)

        # head 1 - mask
        self.up1_mask = up(384, 128)
        self.up2_mask = up(192, 64)
        self.up3_mask = up(128, 64)
        self.up4_mask = up(64, 64, add_shortcut=False)

        # output
        self.out_u = outconv(64, 256)
        self.out_v = outconv(64, 256)
        self.out_w = outconv(64, 256)
        self.out_lat = outconv(256, 3)
        self.out_mask = outconv(64, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Freeze first layers
        _freeze_module(self.conv1)
        _freeze_module(self.bn1)
        _freeze_module(self.layer1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 480 * 640 * 3
        x1 = self.conv1(x)

        # 240 * 320 * 64
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.maxpool(x1)

        # 120 * 160 * 64
        x3 = self.layer1(x2)
        x3 = self.layer2(x3)

        # 60 * 80 * 128
        x4 = self.layer3(x3)

        # head 0 - lat
        x_lat = self.out_lat(x4)
        x_lat = torch.mean(x_lat.view(x_lat.size(0), x_lat.size(1), -1), dim=2)
        lat = project_vecs_onto_sphere(x_lat, 1, True)

        # head 1 - u
        x_u = self.up1_u(x4, x3)
        x_u = self.up2_u(x_u, x2)
        x_u = self.up3_u(x_u, x1)
        x_u = self.up4_u(x_u, x)
        u = self.out_u(x_u)
        u = F.log_softmax(u, dim=1)

        # head 2 - v
        x_v = self.up1_v(x4, x3)
        x_v = self.up2_v(x_v, x2)
        x_v = self.up3_v(x_v, x1)
        x_v = self.up4_v(x_v, x)
        v = self.out_v(x_v)
        v = F.log_softmax(v, dim=1)

        # head 3 - w
        x_w = self.up1_w(x4, x3)
        x_w = self.up2_w(x_w, x2)
        x_w = self.up3_w(x_w, x1)
        x_w = self.up4_w(x_w, x)
        w = self.out_w(x_w)
        w = F.log_softmax(w, dim=1)

        # head 4 - mask
        x_mask = self.up1_mask(x4, x3)
        x_mask = self.up2_mask(x_mask, x2)
        x_mask = self.up3_mask(x_mask, x1)
        x_mask = self.up4_mask(x_mask, x)
        mask = self.out_mask(x_mask)

        # Get probabilities per channel
        sm_hardness = 100
        prob_u = torch.softmax(u * sm_hardness, dim=1)
        prob_v = torch.softmax(v * sm_hardness, dim=1)
        prob_w = torch.softmax(w * sm_hardness, dim=1)

        # Get final colors per channel
        colors = torch.arange(256).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(x.device, x.dtype)
        colors_u = torch.sum(colors * prob_u, dim=1, keepdim=True)
        colors_v = torch.sum(colors * prob_v, dim=1, keepdim=True)
        colors_w = torch.sum(colors * prob_w, dim=1, keepdim=True)

        uvw_sm = torch.cat([colors_u, colors_v, colors_w], dim=1)

        # Get soft mask
        prob_mask = torch.softmax(mask * sm_hardness, dim=1)
        values_mask = torch.arange(2).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(x.device, x.dtype)
        mask_sm = torch.sum(values_mask * prob_mask, dim=1, keepdim=True)

        # Get masked nocs
        uvw_sm_masked = uvw_sm * mask.argmax(dim=1, keepdim=True).expand_as(uvw_sm).float()

        # Form output dict
        output = {}
        output['u'] = u
        output['v'] = v
        output['w'] = w
        output['uvw_sm'] = uvw_sm
        output['uvw_sm_masked'] = uvw_sm_masked
        output['mask'] = mask
        output['mask_sm'] = mask_sm
        output['latent'] = lat

        return output


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def setup_css(pretrained=False, model_path=None, mode='train'):
    """
    Setup CSS network
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        model_path (string): Path to the stored network
        mode ('train'/'eval'): Network mode

    Returns:

    """
    model = resnet18(pretrained=pretrained)
    if model_path:
        model.load_state_dict(torch.load(model_path), strict=True)
        print("CSS net restored.")
    if mode == 'train':
        model.train()
    elif mode == 'eval':
        model.eval()
    return model
