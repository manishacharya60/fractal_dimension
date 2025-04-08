import torch
import torch.nn as nn
import torchvision.models as models

class FractalResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(FractalResNet, self).__init__()
        
        self.resnet = models.resnet18(pretrained=True)
       
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.feature_dim = 512  # For resnet18
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, fractal_dim):
        
        features = self.resnet(x)
        features = features.view(features.size(0), -1)  
        
        x_cat = torch.cat((features, fractal_dim), dim=1)
        out = self.classifier(x_cat)
        return out
