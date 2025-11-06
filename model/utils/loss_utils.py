import torch
from torch import nn
import torch.nn.functional as F
import numpy as np








class BalLossL2(nn.Module):
    def __init__(self, alpha=1, lambda_reg=1):
        super(BalLossL2, self).__init__()
        self.alpha = alpha
        self.lambda_reg = lambda_reg

    def forward(self, probs, labels, model):
        eps = 1e-7
        loss_1 = -1.0 * self.alpha * torch.log(probs + eps) * labels
        loss_0 = -1.0 * torch.log(1 - probs + eps) * (1 - labels)
        classification_loss = torch.mean(loss_0 + loss_1)

        l2_reg = sum(param.pow(2.0).sum() for param in model.parameters())
        total_loss = classification_loss + self.lambda_reg * l2_reg
        return total_loss, classification_loss.item(), self.lambda_reg*l2_reg.item()






