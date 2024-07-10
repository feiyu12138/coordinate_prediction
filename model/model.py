import torch
import torch.nn as nn
import dsntnn
import torchvision.models as models

class FCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)
    
class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations, image_size,backbone='fcn'):
        super().__init__()
        if backbone == 'fcn':
            self.backbone = FCN()
            self.hm_conv = nn.Conv2d(16, n_locations, kernel_size=1, bias=False)
        elif backbone == 'resnet':
            resnet = models.resnet18(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.hm_conv = nn.Conv2d(512, n_locations, kernel_size=1, bias=False)

        self.image_size = image_size

    def forward(self, images):
        # 1. Run the images through our FCN
        feature = self.backbone(images)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(feature)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)

        return coords, heatmaps
    
    