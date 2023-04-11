import torch
import numpy as np
import torch.nn.functional as F

def get_cutmix_and_mixup_output_and_loss(images, target, criterion, model, opts):

    r = np.random.rand(1)
    # default mix_prob : 0.5
    mix_prob = 0.5
    if r < mix_prob:
        switching_prob = np.random.rand(1)
        # Cutmix
        if switching_prob < 0.5:
            slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, opts)
            images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
            if opts.has_auto_encoder:
                output, x_ = model(images)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
                loss += F.mse_loss(x_, images)
            else:
                output = model(images)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
        # Mixup
        else:
            images, y_a, y_b, lam = mixup_data(images, target, opts)
            if opts.has_auto_encoder:
                output, x_ = model(images)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
                loss += F.mse_loss(x_, images)
            else:
                output = model(images)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
    else:
        if opts.has_auto_encoder:
            output, x_ = model(images)
            loss = criterion(output, target)
            loss += F.mse_loss(x_, images)
        else:
            output = model(images)
            loss = criterion(output, target)

    return output, loss


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(x, y, opts):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if opts.mixup_beta > 0:
        lam = np.random.beta(opts.mixup_beta, opts.mixup_beta)
        # lam = np.random.beta(0.8, 0.8)
    else:
        lam = 1

    batch_size = x.size()[0]
    device = x.get_device()
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, opts):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if opts.cutmix_alpha > 0:
        lam = np.random.beta(opts.cutmix_alpha, opts.cutmix_alpha)
        # lam = np.random.beta(1.0, 1.0)
    else:
        lam = 1

    batch_size = x.size()[0]
    device = x.get_device()
    index = torch.randperm(batch_size).to(device)

    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_sliced = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return [bbx1, bby1, bbx2, bby2], y_a, y_b, lam, x_sliced


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)