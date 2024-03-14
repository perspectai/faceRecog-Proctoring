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
        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)

    def forward_once(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

    def forward(self, anchor, positive, negative):
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)
        return anchor_output, positive_output, negative_output


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = F.pairwise_distance(anchor, positive, keepdim=True)
        negative_distance = F.pairwise_distance(anchor, negative, keepdim=True)
        losses = F.relu(positive_distance - negative_distance + self.margin)
        return losses.mean()



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
