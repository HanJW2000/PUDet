import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CCLoss(nn.Module):
    def __init__(self, num_known_classes=20, feat_dim=2, init='random', alpha=0.1, beta=0.9):
        super(CCLoss, self).__init__()
        self.num_known_classes = num_known_classes
        self.feat_dim = feat_dim
        self.beta = beta
        self.alpha = alpha
        assert init in ["random", "fill0", "one-hot","cosproto"]
        if init == 'random':
            self.representatives = nn.Parameter(torch.randn(self.num_known_classes, self.feat_dim))
            nn.init.normal_(self.representatives, std=0.01)
        elif init =='fill0':
            self.representatives = nn.Parameter(torch.Tensor(self.num_known_classes, self.feat_dim))
            self.representatives.data.fill_(0)
        elif init == 'one-hot':
            self.representatives = nn.parameter.Parameter(
                torch.eye(self.num_known_classes, self.feat_dim)
            )
            nn.init.normal_(self.representatives, std=0.01)
        elif init == "cosproto":
            self.representatives = nn.parameter.Parameter(
                torch.Tensor(self.num_known_classes, self.feat_dim)
            )
            nn.init.normal_(self.representatives, std=0.01)

    def dist(self,features, representatives, distance_type):
        if distance_type == 'l1':
            dist = torch.cdist(features, representatives, p=1.0)**2/float(features.shape[1])
        if distance_type == 'l2':
            dist = torch.cdist(features, representatives)**2/float(features.shape[1])
        if distance_type == 'cos':
            dist = features.matmul(representatives.t())
        dist = torch.reshape(dist, [-1, self.num_known_classes, 1])
        dist = torch.mean(dist, dim=2)
        return dist


    def forward(self, features, label, representatives=None, distance_type='hy'):
        if distance_type == 'l1':
            dist = self.dist(features, representatives, "l1")
        elif distance_type == 'l2':
            dist = self.dist(features, representatives, "l2")
        elif distance_type == 'cos':
            representatives = F.normalize(representatives)
            features = F.normalize(features)
            dist = 1 - self.dist(features, representatives,"cos")
        elif distance_type == 'hy':
            cos_dist = self.dist(features, representatives,"cos")
            feat_dist = self.dist(features, representatives, "l2")
            dist = feat_dist - cos_dist
        loss_ce = F.cross_entropy(dist, label)
        return loss_ce






