import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn import CrossEntropyLoss, Module

from utils.utils import args, logger


class Myloss(Module):
    def __init__(self, logit_adj_train, train_dataloader=None, tro_train=None):
        super(Myloss, self).__init__()
        self.loss_function = CrossEntropyLoss()
        self.logit_adj_train = logit_adj_train
        if logit_adj_train:
            self.logit_adjustments = compute_adjustment(train_dataloader, tro_train, args["DEVICE"])
            logger.info(f"logit_adjustments is {self.logit_adjustments}")

    def forward(self, y_pre, y_true):
        if self.logit_adj_train:
            y_pre += self.logit_adjustments
        y_true = y_true.long()
        loss = self.loss_function(y_pre, y_true)

        return loss


def compute_adjustment(train_loader, tro, DEVICE):
    """compute the base probabilities"""
    label_freq = {}
    for i, (inputs, target) in enumerate(train_loader):
        target = target.to(DEVICE)
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(DEVICE)
    return adjustments


def focal_loss(y_pre, y_true):
    y_one_hot = torch.nn.functional.one_hot(y_true, 4)
    loss = torchvision.ops.sigmoid_focal_loss(y_pre, y_one_hot.float(), reduction='mean')
    return loss


class GHM_Loss(nn.Module):
    def __init__(self, bins=10, alpha=0.5):
        '''
        bins: split to n bins
        alpha: hyper-parameter
        '''
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        target = torch.nn.functional.one_hot(target, 4).float()

        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins), device=x.device)
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        return self._custom_loss(x, target, beta[bin_idx])


class GHMC_Loss(GHM_Loss):
    '''
        GHM_Loss for classification
    '''

    def __init__(self, bins, alpha):
        super(GHMC_Loss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target
    