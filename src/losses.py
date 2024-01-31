import torch
import torch.nn as nn


class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2., reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        log_softmax = torch.log_softmax(pred, dim=1)                        # shape=(bs, num_classes)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # collect all log_softmax for all ground truth labels, shape=(bs, 1)
        logpt = logpt.view(-1)                                              # shape=(bs)
        ce_loss = -logpt                                                    # cross_entropy
        pt = torch.exp(logpt)                                               # applying torch.exp to log_softmax, obtain softmax values, shape=(bs)
        focal_loss = (1 - pt) ** self.gamma * ce_loss                       # focal loss, shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        elif self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss