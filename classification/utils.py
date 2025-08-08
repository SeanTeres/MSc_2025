import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import functional as TF
import torch.nn as nn


def to_ordinal_labels(labels, num_classes):
    N = labels.shape[0]
    ordinals = torch.zeros((N, num_classes - 1), dtype=torch.float32, device=labels.device)
    for i in range(1, num_classes):
        ordinals[:, i - 1] = (labels >= i).float()
    return ordinals


def ordinal_preds_to_class(preds, threshold=0.5):
    if preds.max() > 1 or preds.min() < 0:
        preds = torch.sigmoid(preds)
    return (preds > threshold).sum(dim=1)

def reinitialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
