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
# ECA 注意力（通道）
# ------------------------------
class ECABlock(nn.Module):
    """
    Efficient Channel Attention
    建议使用自适应 1D 卷积核大小 k 与通道数 C 相关：
        k = |(log2(C) + b) / gamma|_odd
    """
    def __init__(self, channels, k_size=None, gamma=2, b=1):
        super().__init__()
        if k_size is None:
            k = int(abs((math.log2(channels) + b) / gamma))
            k = k if k % 2 else k + 1
            k = max(3, k)  # 至少 3
        else:
            k = k_size if k_size % 2 == 1 else (k_size + 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x)                        # B,C,1,1
        y = y.squeeze(-1).transpose(-1, -2)   # B,1,C
        y = self.conv(y)                       # B,1,C
        y = y.transpose(-1, -2).unsqueeze(-1) # B,C,1,1
        y = self.act(y)
        return x * y

# ------------------------------
# MobileNetV2 反向残差块（集成 ECA）
# ------------------------------
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_eca=False, eca_kwargs=None):
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

        # 注意力：投影后的 BN 之后、残差相加之前（不破坏 BN 统计）
        self.eca = ECABlock(oup, **(eca_kwargs or {})) if use_eca else None

    def forward(self, x):
        out = self.conv(x)
        if self.eca is not None:
            out = self.eca(out)
        if self.use_res_connect:
            return x + out
        else:
            return out

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.,
                 attn_from_stage=0, attn_kwargs=None):
        """
        仅 ECA 注意力：
        - attn_from_stage: 从第几个 stage 开始给 InvertedResidual 加 ECA（0 基）
        - attn_kwargs: 传给 ECABlock 的字典，如 {'k_size': None}（None 表示自适应核大小）
        """
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
                use_eca_here = (stage_idx >= attn_from_stage)
                self.features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t,
                          use_eca=use_eca_here, eca_kwargs=attn_kwargs)
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
        return model_zoo.load_url(url, model_dir=model_dir)

def mobilenetv2(pretrained=False, attn_kwargs=None, attn_from_stage=0, **kwargs):
    model = MobileNetV2(n_class=1000,
                        attn_from_stage=attn_from_stage,
                        attn_kwargs=attn_kwargs,
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

    model = mobilenetv2(attn_from_stage=0, attn_kwargs={'k_size': None})
    for i, layer in enumerate(model.features):
        print(i, layer)
