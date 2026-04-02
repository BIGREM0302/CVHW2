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
        # TODO:                                                        #
        # Define your CNN model architecture. Note that the first      #
        # input channel is 3, and the output dimension is 10 (class).  #
        ################################################################

        pass

    def forward(self, x):

        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################

        pass
    
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
