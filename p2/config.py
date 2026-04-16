# ============================================================================
# File: config.py
# Date: 2026-03-27
# Author: TA
# Description: Experiment configurations.
# ============================================================================

################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
exp_name = 'semi'  # name of experiment
#exp_name = 'mynet_custom_resnet9'
# Model Options
model_type = 'resnet18'  # 'mynet' or 'resnet18'
#model_type = 'mynet'

# Learning Options
epochs = 100                # train how many epochs 50->100
batch_size = 128            # batch size for dataloader 32->128
use_adam = False           # Adam or SGD optimizer
use_pseudo_labeling = True
#lr = 1e-1                  # learning rate 0.01->0.1
lr = 1e-3
#lr = 3e-4 # avoid gradient explode
#milestones = [16, 32, 45]  # reduce learning rate at 'milestones' epochs
milestones = [50, 75, 90]  # reduce learning raet at 'milesontes' epochs
#milestones = [65, 85]