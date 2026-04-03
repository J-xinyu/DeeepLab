# -*- coding: utf-8 -*-
"""
ResNet50 -> MobileNetV2(PCIR) 知识蒸馏完整示例
- 教师：torchvision.resnet50
- 学生：用户给出的 MobileNetV2(PCIR)（forward 返回 low/high）
- 蒸馏：CE(硬标签) + KD(软标签, KLDiv) + 特征蒸馏(MSE: low_level 对齐 + high_level 对齐)
"""

import math
import os
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from torchvision import datasets, transforms, models

# ===================== 你原始代码（保持逻辑不变） =====================
BatchNorm2d = nn.BatchNorm2d

# ===================== 基础激活 =====================
def act_silu():
    return nn.SiLU(inplace=True)

# ===================== DropPath / Stochastic Depth =====================
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

# ===================== SK-Depthwise（支持空洞；固定 stride=1） =====================
class SKDepthwise(nn.Module):
    """
    并联 3x3 / 5x5 depthwise，softmax 门控融合；固定 stride=1
    现在支持设置 dilation（用于 OS=8/16 时维持感受野）
    """
    def __init__(self, c, dil3: int = 1, dil5: int = 1):
        super().__init__()
        pad3 = dil3
        pad5 = 2 * dil5
        self.dw3 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, pad3, groups=c, bias=False, dilation=dil3),
            BatchNorm2d(c),
            act_silu(),
        )
        self.dw5 = nn.Sequential(
            nn.Conv2d(c, c, 5, 1, pad5, groups=c, bias=False, dilation=dil5),
            BatchNorm2d(c),
            act_silu(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c, 2, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x3 = self.dw3(x)
        x5 = self.dw5(x)
        g = self.pool(x3 + x5)
        a = self.fc(g)
        w = self.softmax(a)
        out = w[:, 0:1] * x3 + w[:, 1:2] * x5
        return out

# ===================== GRN =====================
class GRN(nn.Module):
    """
    Global Response Normalization:
    y = x * (gamma * (||x||/mean||x||)) + beta + x
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)
        return x * (self.gamma * Nx) + self.beta + x

# ===================== 基础卷积 =====================
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),
        act_silu()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        act_silu()
    )

# ===================== PCIR（部分通道倒残差，含 SKDW + GRN；无 SE） =====================
class PCIR(nn.Module):
    """
    inp = a_part + b_part
    a: 1x1 expand -> SKDepthwise(stride=1, 可空洞) -> 1x1 project -> GRN
    b: 恒等旁路（不下采样）
    concat -> 1x1 fuse(stride=stride) 做统一下采样
    """
    def __init__(self, inp, oup, stride, expand_ratio,
                 pc_ratio=0.5, drop_path=0.0,
                 dil3: int = 1, dil5: int = 1):
        super().__init__()
        assert 0 < pc_ratio <= 1.0
        self.stride = stride
        self.part = max(1, int(inp * pc_ratio))
        self.remain = inp - self.part
        hidden = max(8, int(self.part * expand_ratio))

        # a 分支
        a_layers = [
            nn.Conv2d(self.part, hidden, 1, 1, 0, bias=False),
            BatchNorm2d(hidden),
            act_silu(),
            SKDepthwise(hidden, dil3=dil3, dil5=dil5)  # 支持空洞
        ]
        a_layers += [
            nn.Conv2d(hidden, self.part, 1, 1, 0, bias=False),
            BatchNorm2d(self.part),
            GRN(self.part)
        ]
        self.a_path = nn.Sequential(*a_layers)

        # b 分支：不做下采样，保证 concat 前空间尺寸一致
        self.identity_b = (self.remain > 0)

        # 融合：在此统一完成下采样
        self.fuse = nn.Sequential(
            nn.Conv2d(self.part + max(0, self.remain), oup, 1, stride, 0, bias=False),
            BatchNorm2d(oup),
        )

        self.drop_path = DropPath(drop_path)
        self.use_res_connect = (stride == 1 and inp == oup)

    def forward(self, x):
        if self.identity_b and self.remain > 0:
            xa, xb = torch.split(x, [self.part, self.remain], dim=1)
            ya = self.a_path(xa)              # H, W 不变
            y  = torch.cat([ya, xb], dim=1)   # 空间尺寸一致，安全 concat
        else:
            y = self.a_path(x)
        y = self.fuse(y)                      # 在此统一 stride
        if self.use_res_connect:
            y = x + self.drop_path(y)
        return y

# ===================== 主体网络（Deeplab 友好：返回 low_level 与 high_level） =====================
class MobileNetV2(nn.Module):
    """
    与原 MobileNetV2 的层级一致，确保:
    low_level = features[:4](x)  -> 约 1/4 尺度
    x         = features[4:](low_level)
    forward 返回 (low_level, x)，以适配 DeeplabV3+

    额外增强：
    - 支持 output_stride ∈ {8,16,32}（OS 小 → 细节更好 → mIoU/Recall 往往提升）
    - OS 达标后，后续 stage 不再下采样，改用空洞 SKDW 维持感受野
    - widen_last：可选对最后两段适度加宽（提升表示力；默认不加宽）
    - 已移除 SE 模块
    """
    def __init__(self, n_class=1000, input_size=224, width_mult=1.,
                 pc_ratio=0.75, drop_path_rate=0.05,
                 output_stride: int = 32, widen_last: float = 1.0):
        super().__init__()
        assert output_stride in (8, 16, 32), "output_stride 仅支持 8/16/32"
        block = PCIR
        input_channel = 32
        last_channel = 1280

        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # stem 后第1层；累计到2（features[:4] 截取用）
            [6, 24, 2, 2] , # -> /4
            [6, 32, 3, 2] , # -> /8
            [6, 64, 4, 2] , # -> /16
            [6, 96, 3, 1] ,
            [6, 160, 3, 2], # -> /32
            [6, 320, 1, 1] ,
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        # stem (/2)
        self.features = [conv_bn(3, input_channel, 2)]  # /2
        current_os = 2
        dilation = 1

        # 线性递增 DropPath
        total_blocks = sum(x[2] for x in interverted_residual_setting)
        bid = 0

        for stage_idx, (t, c, n, s) in enumerate(interverted_residual_setting):
            # 可选：对最后两段适度加宽
            c_eff = c
            if widen_last != 1.0 and stage_idx >= 5:
                c_eff = int(round(c * widen_last))

            output_channel = int(c_eff * width_mult)

            # 决定该 stage 的 stride（当达到目标 OS 后，后续不再下采样）
            stage_stride = s
            if current_os >= output_stride and s == 2:
                stage_stride = 1
                dilation *= 2

            for i in range(n):
                stride = stage_stride if i == 0 else 1
                dp = drop_path_rate * bid / max(1, total_blocks - 1)
                self.features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride=stride,
                        expand_ratio=t,
                        pc_ratio=0.75,  # 与工厂默认一致
                        drop_path=dp,
                        dil3=dilation,
                        dil5=dilation,
                    )
                )
                input_channel = output_channel
                bid += 1

            if s == 2 and stage_stride == 2:
                current_os *= 2

        # 尾部 1x1
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        # 分类头
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        low_level = self.features[:4](x)   # 约 1/4
        x = self.features[4:](low_level)   # 高层特征
        return low_level, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# ===================== 以下三段保持原样（不修改） =====================
def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url,model_dir=model_dir)

def mobilenetv2(pretrained=False, **kwargs):
    """
    工厂函数：默认即为增强后的配置（可调）：
      - pc_ratio=0.75, drop_path_rate=0.05
      - 可传 output_stride=8/16/32（建议 8 或 16 以提升边界/小目标）
      - 可传 widen_last>1.0 对最后两段适度加宽（如 1.25）
      - 已移除 SE 模块
    """
    defaults = dict(pc_ratio=0.75, drop_path_rate=0.05, output_stride=16, widen_last=1.0)
    defaults.update(kwargs)
    model = MobileNetV2(n_class=1000, **defaults)
    if pretrained:
        model.load_state_dict(
            load_url('https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar'),
            strict=False
        )
    return model

# ===================== 辅助：SyncBN / 冻结 BN =====================
def convert_syncbn(model: nn.Module) -> nn.Module:
    return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

def freeze_bn_stats(model: nn.Module, freeze_affine: bool = True) -> None:
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if freeze_affine:
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

# ===================== 教师网络：ResNet50（返回 low/high/logits） =====================
class ResNet50Teacher(nn.Module):
    """
    low_level: layer1 输出（~1/4）
    high_level: layer4 输出（~1/32）
    logits: 全局池化+fc
    """
    def __init__(self, pretrained=True, num_classes=1000):
        super().__init__()
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        # 提取结构
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)  # /4
        self.layer1 = net.layer1  # /4
        self.layer2 = net.layer2  # /8
        self.layer3 = net.layer3  # /16
        self.layer4 = net.layer4  # /32
        self.avgpool = net.avgpool
        self.fc = nn.Linear(2048, num_classes)
        # 对齐预训练权重到我们的 fc（若类别不同，可随机初始化）
        if pretrained and net.fc.weight.shape[0] == num_classes:
            self.fc.load_state_dict(net.fc.state_dict())

    def forward(self, x):
        x = self.stem(x)            # /4
        low = self.layer1(x)        # /4
        x = self.layer2(low)        # /8
        x = self.layer3(x)          # /16
        high = self.layer4(x)       # /32
        logits = self.fc(torch.flatten(self.avgpool(high), 1))
        return low, high, logits

# ===================== 学生包装器：保持你 forward，不改动学生；补充 logits =====================
class StudentWrapper(nn.Module):
    """
    - backbone: 你的 MobileNetV2，forward -> (low, high)
    - 复用其 pool + classifier 计算 logits（分类蒸馏需要）
    """
    def __init__(self, backbone: MobileNetV2, num_classes: int = 1000):
        super().__init__()
        self.backbone = backbone
        self.pool = backbone.pool
        # 若 classifier 输出维不等于 num_classes，可重置
        in_feats = backbone.classifier[-1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_feats, num_classes),
        )

        # 若你想直接用 backbone 的原 classifier，可注释上面两行并使用：
        # self.classifier = backbone.classifier

    def forward(self, x):
        low, high = self.backbone(x)
        logits = self.classifier(torch.flatten(self.pool(high), 1))
        return {"low": low, "high": high, "logits": logits}

# ===================== 蒸馏损失 =====================
class DistillLoss(nn.Module):
    """
    总损失 = α * CE(student_logits, y) +
           + β * KD_KL(student_logits, teacher_logits, T) +
           + γ * ( MSE(align_low(student_low), teacher_low) + MSE(align_high(student_high), teacher_high) )
    """
    def __init__(self, s_low_c: int, s_high_c: int,
                 t_low_c: int = 256, t_high_c: int = 2048,
                 alpha_ce: float = 1.0, beta_kd: float = 1.0, gamma_feat: float = 1.0,
                 temperature: float = 4.0):
        super().__init__()
        self.alpha = alpha_ce
        self.beta  = beta_kd
        self.gamma = gamma_feat
        self.T = temperature

        # 通道对齐（1x1 conv）
        self.align_low  = nn.Conv2d(s_low_c,  t_low_c,  kernel_size=1, bias=False)
        self.align_high = nn.Conv2d(s_high_c, t_high_c, kernel_size=1, bias=False)

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss(reduction="mean")

    @staticmethod
    def kd_kl(student_logits, teacher_logits, T: float):
        # KLDivLoss 输入需是 log_softmax 与 softmax
        sl = F.log_softmax(student_logits / T, dim=1)
        tl = F.softmax(teacher_logits / T, dim=1)
        loss = F.kl_div(sl, tl, reduction="batchmean") * (T * T)
        return loss

    def forward(self,
                s_feats: Dict[str, torch.Tensor],
                t_feats: Dict[str, torch.Tensor],
                target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        s_low, s_high, s_logits = s_feats["low"], s_feats["high"], s_feats["logits"]
        t_low, t_high, t_logits = t_feats["low"], t_feats["high"], t_feats["logits"]

        # 1) 硬标签 CE
        loss_ce = self.ce(s_logits, target)

        # 2) logit 蒸馏
        loss_kd = self.kd_kl(s_logits, t_logits, self.T)

        # 3) 特征蒸馏（空间自动插值对齐后 MSE）
        # low: /4 尺度；high: 学生可能 /8~/32，统一到教师对应尺度
        def _align_and_mse(s, t, conv_align):
            if s.shape[1] != conv_align.in_channels:
                raise RuntimeError("学生通道不匹配 align conv 设定")
            s_aligned = conv_align(s)
            if s_aligned.shape[-2:] != t.shape[-2:]:
                s_aligned = F.interpolate(s_aligned, size=t.shape[-2:], mode="bilinear", align_corners=False)
            return self.mse(s_aligned, t)

        loss_feat_low  = _align_and_mse(s_low,  t_low,  self.align_low)
        loss_feat_high = _align_and_mse(s_high, t_high, self.align_high)
        loss_feat = loss_feat_low + loss_feat_high

        loss = self.alpha * loss_ce + self.beta * loss_kd + self.gamma * loss_feat
        parts = dict(ce=float(loss_ce.item()),
                     kd=float(loss_kd.item()),
                     feat=float(loss_feat.item()))
        return loss, parts

# ===================== 训练与验证（示例） =====================
def get_imagenet_like_loaders(data_root: str,
                              img_size: int = 224,
                              batch_size: int = 64,
                              num_workers: int = 8):
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*256/224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    # 这里用 ImageFolder 作为例子；实际数据集可替换
    train_set = datasets.FakeData(size=10000, image_size=(3, img_size, img_size),
                                  num_classes=1000, transform=train_tf)
    val_set   = datasets.FakeData(size=2000, image_size=(3, img_size, img_size),
                                  num_classes=1000, transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

@torch.no_grad()
def evaluate(student: StudentWrapper, loader: DataLoader, device: torch.device) -> float:
    student.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = student(x)["logits"]
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

def train_kd(
    epochs: int = 10,
    data_root: str = "./data",
    num_classes: int = 1000,
    img_size: int = 224,
    batch_size: int = 64,
    lr: float = 1e-3,
    alpha_ce: float = 1.0,
    beta_kd: float = 1.0,
    gamma_feat: float = 1.0,
    temperature: float = 4.0,
    output_stride_student: int = 16,
    widen_last: float = 1.0,
    use_amp: bool = True,
    save_path: str = "student_kd_best.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    train_loader, val_loader = get_imagenet_like_loaders(
        data_root=data_root, img_size=img_size, batch_size=batch_size
    )

    # 教师
    teacher = ResNet50Teacher(pretrained=True, num_classes=num_classes).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # 学生（使用你的实现）
    stu_backbone = mobilenetv2(pretrained=False, output_stride=output_stride_student, widen_last=widen_last)
    student = StudentWrapper(stu_backbone, num_classes=num_classes).to(device)

    # 蒸馏损失（学生 low/high 通道来自你网络：low ~ 24ch，high ~ last_channel=1280）
    s_low_c  = student.backbone.features[3][-1].num_features if isinstance(student.backbone.features[3][-1], nn.BatchNorm2d) else 24
    s_high_c = student.backbone.last_channel  # 1280（默认）
    criterion = DistillLoss(
        s_low_c=s_low_c, s_high_c=s_high_c,
        t_low_c=256, t_high_c=2048,
        alpha_ce=alpha_ce, beta_kd=beta_kd, gamma_feat=gamma_feat,
        temperature=temperature
    ).to(device)

    # 优化器
    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        student.train()
        running = {"ce": 0.0, "kd": 0.0, "feat": 0.0}
        for step, (x, y) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # 前向：教师
                # 前向：教师
                t_low, t_high, t_logits = teacher(x)

                # 前向：学生
                s_out = student(x)

                # 组装教师特征
                t_out = {"low": t_low, "high": t_high, "logits": t_logits}

                # 计算蒸馏损失
                loss, parts = criterion(s_out, t_out, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 统计
            for k in running.keys():
                running[k] += parts[k]

            if step % 50 == 0:
                avg = {k: running[k] / step for k in running}
                print(
                    f"Epoch [{epoch}/{epochs}] Step [{step}] "
                    f"CE: {avg['ce']:.4f}  KD: {avg['kd']:.4f}  Feat: {avg['feat']:.4f}"
                )

        # ============== 验证 ==============
        student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device, non_blocking=True)
                vy = vy.to(device, non_blocking=True)
                logits = student(vx)["logits"]
                pred = logits.argmax(dim=1)
                correct += (pred == vy).sum().item()
                total += vy.numel()
        acc = correct / max(1, total)
        print(f"Epoch {epoch} Val Acc: {acc*100:.2f}%")

        # 保存最优
        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "student": student.state_dict(),
                    "epoch": epoch,
                    "acc": best_acc,
                },
                save_path,
            )
            print(f"[Save] Best acc {best_acc*100:.2f}% -> {save_path}")

    print(f"Training done. Best Acc: {best_acc*100:.2f}%")
