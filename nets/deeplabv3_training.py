import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def CE_Loss(inputs, target, cls_weights, num_classes=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice loss
    #--------------------------------------------#
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(
    lr_decay_type,
    lr,             
    min_lr,
    total_iters,    
    warmup_iters_ratio = 0.1,
    warmup_lr_ratio    = 0.1,
    no_aug_iter_ratio  = 0.3,
    step_num           = 10,
    power              = 0.9,   
):
    

    def _warmup_lr(iters, warmup_total_iters, warmup_lr_start, base_lr):
        
        return (base_lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start

    warmup_total_iters  = min(max(int(warmup_iters_ratio * total_iters), 1), max(3, int(0.03 * total_iters)))
    warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
    no_aug_iter         = min(max(int(no_aug_iter_ratio * total_iters), 1), max(15, int(0.05 * total_iters)))

    if lr_decay_type == "poly":

        def poly_lr(base_lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters, power=0.9):
            if iters <= warmup_total_iters:
                return _warmup_lr(iters, warmup_total_iters, warmup_lr_start, base_lr)
            elif iters >= total_iters - no_aug_iter:
                return min_lr
            else:
                t = (iters - warmup_total_iters) / float(max(1, total_iters - warmup_total_iters - no_aug_iter))
                t = min(max(t, 0.0), 1.0)
                return min_lr + (base_lr - min_lr) * pow(1.0 - t, power)

        func = partial(poly_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, power=power)

    elif lr_decay_type == "cos":

        def yolox_warm_cos_lr(base_lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
            if iters <= warmup_total_iters:
                return _warmup_lr(iters, warmup_total_iters, warmup_lr_start, base_lr)
            elif iters >= total_iters - no_aug_iter:
                return min_lr
            else:
                return min_lr + 0.5 * (base_lr - min_lr) * (
                    1.0 + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
                )
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)

    else:

        def step_lr(base_lr, decay_rate, step_size, iters):
            if step_size < 1:
                raise ValueError("step_size must above 1.")
            n = iters // step_size
            out_lr = base_lr * (decay_rate ** n)
            return max(out_lr, min_lr)


        decay_rate  = (min_lr / lr) ** (1 / max(1, (step_num - 1)))
        step_size   = max(1, total_iters // max(1, step_num))
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
