import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np
from detectron2.utils.events import get_event_storage
import torch
import torch.distributions as dists
import torch.nn as nn

class UALoss(nn.Module):
    """Unknown Probability Loss
    """

    def __init__(self,
                 num_classes: int,
                 num_known_classes: int,
                 wi_start_iter: int,
                 loss_type: str = "digamma",
                 evidence: str = "exp",
                 topk: int = 3,
                 lam: float = 1.0,
          ):
        super().__init__()
        self.num_classes = num_classes
        self.num_known_classes = num_known_classes
        self.loss_type = loss_type
        self.evidence = evidence
        self.start_iter = wi_start_iter
        self.topk = topk
        self.alpha = 1.0
        self.lam = lam

    def evidence_func(self, logit):
        if self.evidence == 'relu':
            return F.relu(logit)
        if self.evidence == 'exp':
            return torch.exp(torch.clamp(logit, -10, 10))
        if self.evidence == 'softplus':
            return F.softplus(logit)

    def get_loss_func(self, x):
        if self.loss_type == 'log':
            return torch.log(x)
        elif self.loss_type == 'digamma':
            return torch.digamma(x)
        else:
            raise NotImplementedError

    def topk_sampling(self, scores: Tensor, labels: Tensor):
        fg_inds = labels != self.num_classes
        # fg_inds = labels < self.num_known_classes
        fg_scores, fg_labels = scores[fg_inds], labels[fg_inds]
        bg_scores, bg_labels = scores[~fg_inds], labels[~fg_inds]


        num_fg = fg_scores.size(0)
        topk = num_fg if (self.topk == -1) or (num_fg <
                                               self.topk) else self.topk
        # get the topk
        pos_metric = (fg_scores.size(1)) / torch.sum(self.evidence_func(fg_scores) + 1, dim=1)
        neg_metric = (bg_scores.size(1)) / torch.sum(self.evidence_func(bg_scores) + 1, dim=1)
        _, pos_inds = pos_metric.topk(topk)
        _, neg_inds = neg_metric.topk(topk)
        fg_scores, fg_labels = fg_scores[pos_inds], fg_labels[pos_inds]
        bg_scores, bg_labels = bg_scores[neg_inds], bg_labels[neg_inds]

        return fg_scores, bg_scores, fg_labels, bg_labels

    def _soft_cross_entropy(self, input: Tensor, target: Tensor):
        logprobs = F.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    def forward(self, scores: Tensor, labels: Tensor, ious: Tensor):

        alpha = self.evidence_func(scores) + 1
        S = torch.sum(alpha, dim=1, keepdim=True)  # (B, 1)
        alpha_pred = alpha.detach().clone()  # (N, K)
        uncertainty = (scores.size(1)) / alpha_pred.sum(dim=-1, keepdim=True)  # (N, 1)
        feat_norm = torch.sum(torch.abs(scores), 1).reshape(-1)
        num_sample, num_classes = scores.shape
        mask = torch.arange(num_classes).repeat(
            num_sample, 1).to(scores.device)
        inds = mask != labels[:, None].repeat(1, num_classes)
        ints_new = ~inds
        y = ints_new.long()
        eps = 1e-10
        storage = get_event_storage()
        if storage.iter > self.start_iter:
            grad_norm = torch.sum(torch.abs(1 / alpha_pred - uncertainty) * y,
                                  dim=1)  # sum_j|y_ij * (1/alpha_ij - u_i)|, (N)
            weights = 1.0 /(feat_norm * torch.exp(self.lam * grad_norm) + eps)  # influence-balanced weight
            # weights = 1 / (grad_norm * feat_norm.detach())
            edl_loss = weights * torch.sum(y * (self.get_loss_func(S) - self.get_loss_func(alpha)), dim=1, keepdim=True)
        else:
            edl_loss = torch.sum(y * (self.get_loss_func(S) - self.get_loss_func(alpha)), dim=1, keepdim=True)
        # iou loss
        ious[ious < 0] = eps
        ious_loss = - (ious) * torch.log(1 - uncertainty) - (1 - ious) * torch.log(uncertainty)

        # u loss
        fg_scores, bg_scores, fg_labels, bg_labels = self.topk_sampling(
            scores, labels)

        # sample both fg and bg
        scores = torch.cat([fg_scores, bg_scores])
        labels = torch.cat([fg_labels, bg_labels])

        num_sample, num_classes = scores.shape
        mask = torch.arange(num_classes).repeat(
            num_sample, 1).to(scores.device)
        inds = mask != labels[:, None].repeat(1, num_classes)
        mask = mask[inds].reshape(num_sample, num_classes - 1)

        gt_scores = torch.gather(
            F.softmax(scores, dim=1), 1, labels[:, None]).squeeze(1)
        mask_scores = torch.gather(scores, 1, mask)

        gt_scores[gt_scores < 0] = 0.0
        targets = torch.zeros_like(mask_scores)

        num_fg = fg_scores.size(0)

        targets[:num_fg, self.num_classes-2] = gt_scores[:num_fg] * \
            (1-gt_scores[:num_fg]).pow(self.alpha)
        targets[num_fg:, self.num_classes-1] = gt_scores[num_fg:] * \
            (1-gt_scores[num_fg:]).pow(self.alpha)
        ual_loss = self._soft_cross_entropy(mask_scores, targets.detach())


        return edl_loss.mean(), ious_loss.mean(), ual_loss