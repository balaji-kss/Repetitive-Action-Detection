import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
from torchvision import models
import torch

class ImageActNet(nn.Module):
    def __init__(self, inp_channels=3, num_classes=1):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(inp_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):

        x = self.model(x)

        return x

class ImagePoseActNet(nn.Module):

    # Task Boundary Detection Network

    def __init__(self, inp_channels=3, num_classes=1):
        super().__init__()

        # Image Feature Extractor
        self.pretrained = models.resnet18(pretrained=True)
        self.pretrained.conv1 = nn.Conv2d(inp_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        features = nn.ModuleList(self.pretrained.children())[:-1]
        self.pretrained = nn.Sequential(*features)

        # Pose Feature Extractor
        self.conv1 = nn.Conv1d(30, 256, 1, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=256, eps=1e-5, affine=True)

        self.conv2 = nn.Conv1d(256, 256, 1, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=256, eps=1e-5, affine=True)

        self.conv3 = nn.Conv1d(256, 256, 1, stride=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=256, eps=1e-5, affine=True)
        
        self.fc1 = nn.Linear(768, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

    def forward(self, img, joints):
        
        # Image features
        bz = img.shape[0]
        features = self.pretrained(img)
        features = features.view(bz,-1)

        # Pose features
        p_out = self.relu(self.bn1(self.conv1(joints))) # [256, 5]
        p_out = self.relu(self.bn2(self.conv2(p_out)))  # [256, 7]
        p_out = self.relu(self.bn3(self.conv3(p_out)))  # [256, 3]
        p_out = p_out.view(bz,-1)  #flatten 768

        p_out = self.relu(self.fc1(p_out))
        p_out = self.relu(self.fc2(p_out))

        concat = torch.cat((features, p_out), 1)
        
        # Fuse image and pose features
        out = self.relu(self.fc3(concat))
        out = self.relu(self.fc4(out))
        out = self.relu(self.fc5(out))

        return out

if __name__ == "__main__":

    model = ImagePoseActNet(num_classes=1)
    print('model ', model)
    