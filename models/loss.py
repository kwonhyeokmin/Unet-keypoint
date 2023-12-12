from torch.nn.modules.loss import _Loss
import torch


class UNetCrossEntropyLoss(_Loss):
    def __init__(self):
        super(UNetCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = -torch.sum(torch.mul(y_true, torch.log(y_pred)))
        return loss