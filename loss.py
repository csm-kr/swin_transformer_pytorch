import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy


def build_loss(opts):
    if opts.is_vit_data_augmentation:
        criterion = LabelSmoothingCrossEntropy()  # default smoothing 0.1
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion