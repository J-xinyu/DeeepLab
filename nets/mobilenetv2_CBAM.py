import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

BatchNorm2d = nn.BatchNorm2d

# ------------------------------
# 基础卷积块
# ------------------------------
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# ------------------------------
# CBAM 注意力模块（通道 + 空间）
# ------------------------------
class ChannelAttention(nn.Module):
    """CBAM 通道注意力：avg/max 池化 -> 共享 MLP -> 相加 -> Sigmoid"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        out = avg_out + max_out
        w = self.sigmoid(out)
        return x * w

class SpatialAttention(nn.Module):
    """CBAM 空间注意力：沿通道做 avg/max -> 拼接 -> 7x7 Conv -> Sigmoid"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg, mx], dim=1)  # [B,2,H,W]
        y = self.conv(y)
        w = self.sigmoid(y)
        return x * w

class CBAM(nn.Module):
    """CBAM: 通道注意力(MLP) -> 空间注意力"""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ------------------------------
# MobileNetV2 反向残差块（集成 CBAM）
# ------------------------------
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_cbam=False, cbam_kwargs=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        # 仅当步长为1且通道不变时使用残差
        self.use_res_connect = (self.stride == 1 and inp == oup)

        if expand_ratio == 1:
            # 深度可分离卷积（DW）+ 线性投影
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            # PW 扩张 -> DW -> PW 线性投影
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

        # 注意力放置：投影后的 BN 之后、残差相加之前（不破坏 BN 统计）
        self.cbam = CBAM(oup, **(cbam_kwargs or {})) if use_cbam else None

    def forward(self, x):
        out = self.conv(x)
        if self.cbam is not None:
            out = self.cbam(out)
        if self.use_res_connect:
            return x + out
        else:
            return out

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.,
                 attn_from_stage=0, cbam_kwargs=None):

        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        # t: expand_ratio, c: out_channels, n: repeats, s: stride
        interverted_residual_setting = [
            [1, 16, 1, 1],  # stage 0
            [6, 24, 2, 2],  # stage 1
            [6, 32, 3, 2],  # stage 2
            [6, 64, 4, 2],  # stage 3
            [6, 96, 3, 1],  # stage 4
            [6, 160, 3, 2], # stage 5
            [6, 320, 1, 1], # stage 6
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        # Stem: 3->32, /2
        self.features = [conv_bn(3, input_channel, 2)]

        stage_idx = 0
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                use_cbam_here = (stage_idx >= attn_from_stage)
                self.features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t,
                          use_cbam=use_cbam_here, cbam_kwargs=cbam_kwargs)
                )
                input_channel = output_channel
            stage_idx += 1

        # 最后 1x1 卷积
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)  # 全局平均池化
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return torch.hub.load_state_dict_from_url(url, model_dir=model_dir, map_location=map_location)

def mobilenetv2(pretrained=False, cbam_kwargs=None, attn_from_stage=0, **kwargs):
    
    model = MobileNetV2(n_class=1000,
                        attn_from_stage=attn_from_stage,
                        cbam_kwargs=cbam_kwargs,
                        **kwargs)
    if pretrained:
        state = load_url(
            'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar'
        )
        model.load_state_dict(state, strict=False)  
    return model

# ------------------------------
# 调试
# ------------------------------
if __name__ == "__main__":
    # 示例：启用 CBAM，且从 stage 0 开始启用
    model = mobilenetv2(attn_from_stage=0, cbam_kwargs={'reduction': 16, 'spatial_kernel': 7})
    for i, layer in enumerate(model.features):
        print(i, layer)
