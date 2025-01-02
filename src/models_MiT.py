import os
from functools import partial
import requests
from urllib.parse import urljoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def downsample_basic_block(x, planes, stride):
    # downsample depth/width/height based on stride
    out = F.avg_pool3d(x, kernel_size=1, stride=stride) 
    # add channels
    zero_pads = torch.Tensor(         
        out.size(0), planes - out.size(1),
        out.size(2), out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = torch.cat([out.data, zero_pads], dim=1)
    return out


class BasicBlock(nn.Module):
    expansion = 1
    Conv3d = staticmethod(conv3x3x3)

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self.Conv3d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.Conv3d(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
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
    Conv3d = nn.Conv3d

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = self.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = self.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = self.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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


class ResNet3D(nn.Module):

    Conv3d = nn.Conv3d

    def __init__(self, block, layers, shortcut_type='B', num_classes=305):
        self.inplanes = 64
        super(ResNet3D, self).__init__()
        self.conv1 = self.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        self.dropout = nn.Dropout(p=0.5)  # Add dropout here
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.init_weights()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    self.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, self.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def modify_resnets(model):
    # Modify attributs
    model.last_linear, model.fc = model.fc, None

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    setattr(model.__class__, 'features', features)
    setattr(model.__class__, 'logits', logits)
    setattr(model.__class__, 'forward', forward)
    return model


ROOT_URL = 'http://moments.csail.mit.edu/moments_models/'
weights = {
    'resnet50': 'moments_v2_RGB_resnet50_imagenetpretrained.pth.tar',
    'resnet3d50': 'moments_v2_RGB_imagenet_resnet3d50_segment16.pth.tar',
    'multi_resnet3d50': 'multi_moments_v2_RGB_imagenet_resnet3d50_segment16.pth.tar',
}


def download_file(url, save_path):
    """Download a file from a URL."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def load_checkpoint(weight_file):
    # Define the target directory for pretrained models
    model_dir = os.path.join(os.getcwd(),'model_checkpoints','pretrained')
    os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn’t exist

    # Define the full path where the weight file will be saved
    weight_path = os.path.join(model_dir, weight_file)

    # Check if the weight file exists locally
    if not os.path.isfile(weight_path) or not os.access(weight_path, os.R_OK):
        # Construct the full URL for the weight file
        weight_url = urljoin(ROOT_URL, weight_file)
        download_file(weight_url, weight_path)
    checkpoint = torch.load(weight_path, map_location=lambda storage, loc: storage, weights_only=False)  # Load on cpu
    return {str.replace(str(k), 'module.', ''): v for k, v in checkpoint['state_dict'].items()}



def resnet50(num_classes=305, pretrained=True):
    model = models.__dict__['resnet50'](num_classes=num_classes)
    if pretrained:
        model.load_state_dict(load_checkpoint(weights['resnet50']))
    model = modify_resnets(model)
    return model


def resnet3d50(num_classes=305, pretrained=True, **kwargs):
    """Constructs a ResNet3D-50 model."""
    model = modify_resnets(ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs))
    if pretrained:
         model.load_state_dict(load_checkpoint(weights['resnet3d50']))
    return model


def multi_resnet3d50(num_classes=292, pretrained=True, **kwargs):
    """Constructs a ResNet3D-50 model."""
    model = modify_resnets(ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs))
    if pretrained:
        model.load_state_dict(load_checkpoint(weights['multi_resnet3d50']))
    return model


def load_model(arch):
    model = {'resnet3d50': resnet3d50,
             'multi_resnet3d50': multi_resnet3d50, 'resnet50': resnet50}.get(arch, 'resnet3d50')()
    model.eval()
    return model


def load_transform():
    """Load the image transformer."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])


def load_categories(filename):
    """Load categories."""
    with open(filename) as f:
        return [line.rstrip() for line in f.readlines()]

# Modify model to have multi-head output

class MultiHeadLinear(nn.Module):
    def __init__(self, in_features):
        super(MultiHeadLinear, self).__init__()
        self.dropout = nn.Dropout(p=0.5)  # Add a dropout layer
        self.distance_score = nn.Linear(in_features, 1)
        self.distance_confidence = nn.Linear(in_features, 1)
        self.object_score = nn.Linear(in_features, 1)
        self.object_confidence = nn.Linear(in_features, 1)
        self.expanse_score = nn.Linear(in_features, 1)
        self.expanse_confidence = nn.Linear(in_features, 1)
        self.facingness_score = nn.Linear(in_features, 1)
        self.facingness_confidence = nn.Linear(in_features, 1)
        self.communicating_score = nn.Linear(in_features, 1)
        self.communicating_confidence = nn.Linear(in_features, 1)
        self.joint_score = nn.Linear(in_features, 1)
        self.joint_confidence = nn.Linear(in_features, 1)
        self.valence_score = nn.Linear(in_features, 1)
        self.valence_confidence = nn.Linear(in_features, 1)
        self.arousal_score = nn.Linear(in_features, 1)
        self.arousal_confidence = nn.Linear(in_features, 1)
        self.location_score = nn.Linear(in_features, 1)
        self.location_confidence = nn.Linear(in_features, 1)
        self.peoplecount = nn.Linear(in_features, 5)
        self.peoplecount_certain = nn.Linear(in_features, 1)
        
    def forward(self, x):

        # Compute scores and confidence values
        distance_score = self.distance_score(x)  
        distance_confidence = self.distance_confidence(x)
        object_score = self.object_score(x)
        object_confidence = self.object_confidence(x)
        expanse_score = self.expanse_score(x)
        expanse_confidence = self.expanse_confidence(x)
        facingness_score = self.facingness_score(x)
        facingness_confidence = self.facingness_confidence(x)
        communicating_score = self.communicating_score(x)
        communicating_confidence = self.communicating_confidence(x)
        joint_score = self.joint_score(x)
        joint_confidence = self.joint_confidence(x)
        valence_score = self.valence_score(x)
        valence_confidence = self.valence_confidence(x)
        arousal_score = self.arousal_score(x)
        arousal_confidence = self.arousal_confidence(x)
        location_score = self.location_score(x)
        location_confidence = self.location_confidence(x)
        peoplecount = self.peoplecount(x)
        peoplecount_certain = self.peoplecount_certain(x)

        # Aggregate outputs
        outputs = {
            "distance_score": distance_score,
            "distance_confidence": distance_confidence,
            "object_score": object_score,
            "object_confidence": object_confidence,
            "expanse_score": expanse_score,
            "expanse_confidence": expanse_confidence,
            "facingness_score": facingness_score,
            "facingness_confidence": facingness_confidence,
            "communicating_score": communicating_score,
            "communicating_confidence": communicating_confidence,
            "joint_score": joint_score,
            "joint_confidence": joint_confidence,
            "valence_score": valence_score,
            "valence_confidence": valence_confidence,
            "arousal_score": arousal_score,
            "arousal_confidence": arousal_confidence,
            "location_score": location_score,
            "location_confidence": location_confidence,
            "peoplecount": peoplecount,
            "peoplecount_certain": peoplecount_certain
        }

        return outputs





