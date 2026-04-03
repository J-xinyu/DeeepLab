import torch
import torch.nn as nn
import torch.nn.functional as F
#from nets.xception import xception
from nets.mobilenetv3_basic import mobilenetv3
from functools import partial

class MobileNetV3(nn.Module):
    """
    MobileNetV3 语义分割骨干网络（基于torchvision官方实现）
    支持Large和Small两种模式，支持downsample_factor=8/16
    """
    def __init__(self, mode='large', downsample_factor=16, pretrained=True):
        super(MobileNetV3, self).__init__()
        assert downsample_factor in [8, 16]

        # 加载官方预训练模型
       
        model = mobilenetv3(pretrained=pretrained)
        # Small模式下stride=2的层索引:
        # features[0](stem), [1](56x56), [3](28x28), [8](14x14)
        self.features = model.features[:-1]  # 去掉最后的conv
        self.down_idx = [1, 3, 8]  # Small只有3个下采样阶段（从56到7）

        self.total_idx = len(self.features)

        # 应用空洞卷积策略（与MobileNetV2完全一致）
        if downsample_factor == 8:
            # 对最后两个下采样层应用dilation
            # 倒数第二个：dilate=2，最后一个：dilate=4
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            # 只对最后一个下采样层应用dilation=2
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        """
        将stride=2的卷积改为stride=1，并应用空洞卷积
        支持3x3和5x5卷积核（MobileNetV3 Large使用5x5）
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # 处理 stride=2 -> stride=1 的情况
            if m.stride == (2, 2) or m.stride == 2:
                m.stride = (1, 1)
                # 只处理非1x1卷积
                if m.kernel_size[0] > 1:
                    # 目标dilation rate（对于factor=8的最后一层，dilate=4，这里用一半）
                    new_dilate = dilate // 2
                    m.dilation = (new_dilate, new_dilate)
                    # padding = dilation * (kernel_size - 1) // 2
                    # 对于3x3: padding=dilate//2, 对于5x5: padding=dilate
                    m.padding = (new_dilate * (m.kernel_size[0] - 1) // 2,
                               new_dilate * (m.kernel_size[1] - 1) // 2)
            else:
                # 原本stride=1的层，直接应用dilation
                if m.kernel_size[0] > 1:
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate * (m.kernel_size[0] - 1) // 2,
                               dilate * (m.kernel_size[1] - 1) // 2)

    def forward(self, x):
        """
        前向传播，返回low_level_features和high_level_features
        low_level_features: stride=4的特征（128x128 for 512 input），用于DeepLabV3+ decoder
        high_level_features: 最终特征
        """
        # 取前4层作为low_level_features（对应stride=4，与MobileNetV2一致）
        # MobileNetV3: 
        #   features[0]: stride=2 (256x256)
        #   features[1]: stride=1 (256x256) 
        #   features[2]: stride=2 (128x128)
        #   features[3]: stride=1 (128x128) <- 这里取到idx=3，输出128x128
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


#-----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
#-----------------------------------------#
class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenetv3", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone=="xception":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenetv3":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV3(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 96
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #-----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
        #----------------------------------#
        #   浅层特征边
        #----------------------------------#
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
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        
        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

