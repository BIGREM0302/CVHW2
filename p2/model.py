# ============================================================================
# File: model.py
# Date: 2026-03-27
# Author: TA
# Description: Model architecture.
# ============================================================================

import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module): 
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
    
if __name__ == '__main__':
    model = ResNet18()
    print(model)
