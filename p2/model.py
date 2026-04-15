# ============================================================================
# File: model.py
# Date: 2026-03-27
# Author: TA
# Description: Model architecture.
# ============================================================================

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SPPF(nn.Module):
    # 靈感來自 YOLOv5 的 Spatial Pyramid Pooling - Fast
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        # 先降維一半，減少後續拼接時的計算量
        c_hidden = in_channels // 2  
        
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels, c_hidden, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.SiLU(inplace=True)
        )
        
        # 拼接 4 個特徵圖後 (1個原始 + 3個池化)，通道數會變成 c_hidden * 4，再把它轉回 out_channels
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_hidden * 4, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # 使用 padding 保持長寬不變
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        # 連續過池化層，感受野等效於 5x5, 9x9, 13x13 (若 kernel=5)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        
        # 在通道維度 (dim=1) 將不同感受野的特徵拼接起來
        return self.cv2(torch.cat((x, y1, y2, y3), 1))

# ==========================================
# 1. Channel Attention
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        # Squeeze: 降維
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        # Excitation: 升維回原通道數
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        # Global Average Pooling
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        # 將權重乘回原特徵圖
        return x * y

# ==========================================
# 2. Inverted Bottleneck + SE
# ==========================================
class ModernBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4):
        super(ModernBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        
        # 只有在輸入輸出維度一致且 stride=1 時才使用殘差連接
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        # (1) Pointwise Expand (升維)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True) # 使用更現代的 SiLU (Swish)
            ])
        
        # (2) Depthwise Convolution (逐通道卷積)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, 
                      groups=hidden_dim, bias=False), # groups=hidden_dim 是 Depthwise 的關鍵
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])

        # (3) 插入注意力機制
        layers.append(SEBlock(hidden_dim))

        # (4) Pointwise Project (降維，且不加激活函數，即 Linear Bottleneck)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MyNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyNet, self).__init__()
        
        # Stem (準備層) - Input: 3 x 32 x 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # 堆疊 Modern Blocks
        self.blocks = nn.Sequential(
            # in_channels, out_channels, stride, expand_ratio
            ModernBlock(32, 32, stride=1, expand_ratio=1),      # 輸出: 32 x 32 x 32
            
            ModernBlock(32, 64, stride=2, expand_ratio=4),      # 輸出: 64 x 16 x 16 (Downsample)
            ModernBlock(64, 64, stride=1, expand_ratio=4),
            
            ModernBlock(64, 128, stride=2, expand_ratio=4),     # 輸出: 128 x 8 x 8 (Downsample)
            ModernBlock(128, 128, stride=1, expand_ratio=4),
            
            ModernBlock(128, 256, stride=2, expand_ratio=4),    # 輸出: 256 x 4 x 4 (Downsample)
            ModernBlock(256, 256, stride=1, expand_ratio=4)
        )
        
        #self.sppf = SPPF(256, 256, kernel_size=5)
        # 分類器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 全域平均池化 (現代架構多用 Avg 而非 Max)
            nn.Flatten(),
            nn.Dropout(0.2),         # 稍微降低 Dropout，因為 Depthwise 已經有正規化效果
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        #x = self.sppf(x)
        x = self.classifier(x)
        return x

class _MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
        
        ################################################################
        # TODO: Define your CNN model architecture.
        ################################################################
        
        # 定義一個好用的卷積區塊 (Conv -> BatchNorm -> ReLU -> [MaxPool])
        def conv_block(in_channels, out_channels, pool=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        # Input: 3 x 32 x 32
        self.prep = conv_block(3, 64) # 輸出: 64 x 32 x 32
        
        # Layer 1
        self.layer1 = conv_block(64, 128, pool=True) # 輸出: 128 x 16 x 16
        self.res1 = nn.Sequential(
            conv_block(128, 128),
            conv_block(128, 128)
        )
        
        # Layer 2
        self.layer2 = conv_block(128, 256, pool=True) # 輸出: 256 x 8 x 8
        
        # Layer 3
        self.layer3 = conv_block(256, 512, pool=True) # 輸出: 512 x 4 x 4
        self.res2 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), # 全域池化，輸出: 512 x 1 x 1
            nn.Flatten(),
            nn.Dropout(0.3),         # 加入 Dropout 避免過擬合
            nn.Linear(512, 10)       # 分類到 10 個類別
        )

    def forward(self, x):

        ##########################################
        # TODO: Define the forward path of your model.
        ##########################################
        
        # 準備層
        out = self.prep(x)
        
        # Block 1 (包含 Residual Connection)
        out = self.layer1(out)
        out = self.res1(out) + out 
        
        # Block 2
        out = self.layer2(out)
        
        # Block 3 (包含 Residual Connection)
        out = self.layer3(out)
        out = self.res2(out) + out

        # 分類器
        out = self.classifier(out)
        return out
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        # try to load the pretrained weights
        self.resnet = models.resnet18(weights=None)  # Python3.8 w/ torch 2.2.1
        # self.resnet = models.resnet18(pretrained=False)  # Python3.6 w/ torch 1.10.1
        # (batch_size, 512)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        # (batch_size, 10)

        #######################################################################
        # TODO (optional):                                                     #
        # Some ideas to improve accuracy if you can't pass the strong         #
        # baseline:                                                           #
        #   1. reduce the kernel size, stride of the first convolution layer. # 
        #   2. remove the first maxpool layer (i.e. replace with Identity())  #
        # You can run model.py for resnet18's detail structure                #
        #######################################################################
        # 1. 縮小第一層卷積的 kernel size 與 stride
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 2. 拔除 MaxPool 層 (替換成 Identity) 以保留特徵圖解析度
        self.resnet.maxpool = nn.Identity()
        ############################## TODO End ###############################

    def forward(self, x):
        return self.resnet(x)
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    #model = ResNet18()
    model = MyNet()
    print(model)

    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
