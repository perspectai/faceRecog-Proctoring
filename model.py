import torch
import torch.nn.functional as F
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1



class SiameseNetwork(nn.Module):
    def __init__(self, pretrained=True):
        super(SiameseNetwork, self).__init__()
        # Load a pre-trained InceptionResNetV1 model
        self.backbone = InceptionResnetV1(pretrained='vggface2' if pretrained else None)
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512,512)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)

    def forward_once(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        #x = self.batchnorm1(x)
        x = self.fc2(x)
        #x = self.relu2(x)
        return x

    def forward(self, img1, img2):
        output1 = self.forward_once(img1)
        output2 = self.forward_once(img2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive