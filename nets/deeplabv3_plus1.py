import torch
import torch.nn as nn
import torch.nn.functional as F
#from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x 


# ----------------------------------------- #
# DenseASPP 特征提取模块
# 采用密集连接的空洞卷积形成连续多尺度上下文
# rates 会根据输出步长(16或8)做自适应缩放
# ----------------------------------------- #
class DenseASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rates=(3, 6, 12, 18, 24), growth_rate=64, bn_mom=0.1):
        super(DenseASPP, self).__init__()
        self.blocks = nn.ModuleList()
        in_ch = dim_in
        for d in rates:
            block = nn.Sequential(
                nn.Conv2d(in_ch, growth_rate, kernel_size=3, stride=1,
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(growth_rate, momentum=bn_mom),
                nn.ReLU(inplace=True),
            )
            self.blocks.append(block)
            in_ch += growth_rate  # dense 连接：通道数逐层累加

        # 最终投影到 dim_out（与原 ASPP 的 256 对齐）
        self.project = nn.Sequential(
            nn.Conv2d(in_ch, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = [x]
        out = x
        for blk in self.blocks:
            new_feat = blk(out)
            feats.append(new_feat)
            out = torch.cat(feats, dim=1)
        out = self.project(out)
        return out


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenetv2", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone=="xception":
            # 浅层特征 [128,128,256]；主干输出 [30,30,2048]
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenetv2":
            # 浅层特征 [128,128,24]；主干输出 [30,30,320]
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        # ----------------------------------------- #
        # 用 DenseASPP 替换 ASPP
        # base_rate = 16 // downsample_factor
        # 例如 OS=16 时保持 (3,6,12,18,24)
        #     OS=8  时放大一倍 (6,12,24,36,48)
        # ----------------------------------------- #
        base_rate = 16 // downsample_factor
        dense_rates = tuple([r * base_rate for r in (3, 6, 12, 18, 24)])
        self.aspp = DenseASPP(dim_in=in_channels, dim_out=256, rates=dense_rates, growth_rate=64, bn_mom=0.1)
        
        # ---------------------------------- #
        # 浅层边分支
        # ---------------------------------- #
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )		

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # 两个特征层：浅层 + 主干输出
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        
        # 上采样并与浅层特征融合
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)),
                          mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x