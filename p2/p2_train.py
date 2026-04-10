# ============================================================================
# File: p2_train.py
# Date: 2026-03-27
# Author: TA
# Description: Training a model and save the best model.
# ============================================================================

import os
import sys  
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

import config as cfg
from model import MyNet, ResNet18
from dataset import get_dataloader
from utils import set_seed, write_config_log, write_result_log

import torch.nn.functional as F
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        # 確保回傳格式跟 CIFAR10Dataset 一模一樣 (dict)
        return {
            'images': self.images[idx],
            'labels': self.labels[idx]
        }

def generate_pseudo_labels(teacher_model, unlabeled_loader, device, threshold=0.95):
    """
    使用訓練好的 teacher_model 為無標籤資料產生 Pseudo-labels。
    只保留信心水準大於 threshold 的預測結果。
    """
    teacher_model.eval()
    pseudo_images = []
    pseudo_labels = []
    
    print("Generating pseudo-labels...")
    with torch.no_grad():
        for data in unlabeled_loader:
            images = data['images'].to(device)
            logits = teacher_model(images)
            
            # 將 logits 轉成機率
            probs = F.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            # 找出信心水準大於門檻的資料
            mask = max_probs > threshold
            if mask.sum() > 0:
                pseudo_images.append(images[mask].cpu())
                pseudo_labels.append(preds[mask].cpu())
                
    if len(pseudo_images) == 0:
        print("No confident pseudo-labels generated.")
        return None
        
    pseudo_images = torch.cat(pseudo_images, dim=0)
    pseudo_labels = torch.cat(pseudo_labels, dim=0)
    print(f"Generated {len(pseudo_labels)} high-confidence pseudo-labels.")
    
    # 修改這裡：使用自定義類別
    return PseudoDataset(pseudo_images, pseudo_labels)

def plot_learning_curve(
        logfile_dir: str,
        result_lists: dict
    ):

    epochs = range(1, len(result_lists['train_acc']) + 1)

    # 1️⃣ Training Accuracy
    plt.figure()
    plt.plot(epochs, result_lists['train_acc'], label='Train Acc')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(logfile_dir, 'train_acc.png'))
    plt.close()

    # 2️⃣ Validation Accuracy
    plt.figure()
    plt.plot(epochs, result_lists['val_acc'], label='Val Acc')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(logfile_dir, 'val_acc.png'))
    plt.close()

    # 3️⃣ Training Loss
    plt.figure()
    plt.plot(epochs, result_lists['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(logfile_dir, 'train_loss.png'))
    plt.close()

    # 4️⃣ Validation Loss
    plt.figure()
    plt.plot(epochs, result_lists['val_loss'], label='Val Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(logfile_dir, 'val_loss.png'))
    plt.close()

def train(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        logfile_dir: str,
        model_save_dir: str,
        criterion: nn.Module,
        optimizer: torch.optim,
        scheduler: torch.optim,
        device: torch.device
    ):
    '''
    Training and validation process.
    - Args:
        - model: nn.Module, the model to be trained.
        - train_loader: DataLoader, the dataloader of training set.
        - val_loader: DataLoader, the dataloader of validation set.
        - logfile_dir: str, the directory to save the log files.
        - model_save_dir: str, the directory to save the best model.
        - criterion: nn.Module, the loss function.
        - optimizer: torch.optim, the optimizer.
        - scheduler: torch.optim.lr_scheduler, the learning rate scheduler.
        - device: torch.device, the device to run the model.
    - Returns:
        - None
    '''

    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        ##### TRAINING #####
        train_start_time = time.time()
        train_loss = 0.0
        train_correct = 0.0
        model.train()
        for batch, data in enumerate(train_loader):
            sys.stdout.write(f'\r[{epoch + 1}/{cfg.epochs}] Train batch: {batch + 1} / {len(train_loader)}')
            sys.stdout.flush()
            # Data loading. (batch_size, 3, 32, 32), (batch_size)
            images, labels = data['images'].to(device), data['labels'].to(device)
            # Forward pass. input: (batch_size, 3, 32, 32), output: (batch_size, 10)
            pred = model(images)
            # Calculate loss.
            loss = criterion(pred, labels)
            # Backprop. (update model parameters)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Evaluate.
            train_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
            train_loss += loss.item()
        # Print training result
        train_time = time.time() - train_start_time
        train_acc = train_correct / len(train_loader.dataset)
        train_loss /= len(train_loader)
        train_acc_list.append(train_acc.cpu().numpy())
        train_loss_list.append(train_loss)
        print()
        print(f'[{epoch + 1}/{cfg.epochs}] {train_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Train Loss: {train_loss:.5f}')

        ##### VALIDATION #####
        model.eval()
        with torch.no_grad():
            val_start_time = time.time()
            val_loss = 0.0
            val_correct = 0.0
            #############################################################
            # TODO:                                                     #
            # Finish forward part in validation, you can refer to the   #
            # training part.                                            #
            #                                                           #
            # NOTE:                                                     #
            # You don't have to update parameters, just record the      #
            # accuracy and loss.                                        #
            #############################################################
            for data in val_loader:
                images = data['images'].to(device)
                labels = data['labels'].to(device)
                
                pred = model(images)
                loss = criterion(pred, labels)
                
                val_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
                val_loss += loss.item()
            ######################### TODO End ##########################

        # Print validation result
        val_time = time.time() - val_start_time
        val_acc = val_correct / len(val_loader.dataset)
        val_loss /= len(val_loader)
        val_acc_list.append(val_acc.cpu().numpy())
        val_loss_list.append(val_loss)
        print()
        print(f'[{epoch + 1}/{cfg.epochs}] {val_time:.2f} sec(s) Val Acc: {val_acc:.5f} | Val Loss: {val_loss:.5f}')
        
        # Scheduler step
        scheduler.step()

        ##### WRITE LOG #####
        is_better = val_acc >= best_acc
        epoch_time = train_time + val_time
        write_result_log(os.path.join(logfile_dir, 'result_log.txt'),
                         epoch, epoch_time,
                         train_acc, val_acc,
                         train_loss, val_loss,
                         is_better)

        ##### SAVE THE BEST MODEL #####
        if is_better:
            print(f'[{epoch + 1}/{cfg.epochs}] Save best model to {model_save_dir} ...')
            torch.save(model.state_dict(),
                       os.path.join(model_save_dir, 'model_best.pth'))
            best_acc = val_acc

        ##### PLOT LEARNING CURVE #####
        ##### TODO: check plot_learning_curve() in this file #####
        current_result_lists = {
            'train_acc': train_acc_list,
            'train_loss': train_loss_list,
            'val_acc': val_acc_list,
            'val_loss': val_loss_list
        }
        plot_learning_curve(logfile_dir, current_result_lists)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', 
                        help='dataset directory', 
                        type=str, 
                        default='../hw2_data/p2_data/')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    # Experiment name
    exp_name = cfg.model_type \
        + datetime.now().strftime('_%Y_%m_%d_%H_%M_%S') \
        + '_' + cfg.exp_name

    # Write log file for config
    logfile_dir = os.path.join('./experiment', exp_name, 'log')
    os.makedirs(logfile_dir, exist_ok=True)
    write_config_log(os.path.join(logfile_dir, 'config_log.txt'))

    # Fix a random seed for reproducibility
    set_seed(2025)

    # Check if GPU is available, otherwise CPU is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    ##### MODEL #####
    ##### TODO: check model.py #####
    model_save_dir = os.path.join('./experiment', exp_name, 'model')
    os.makedirs(model_save_dir, exist_ok=True)

    if cfg.model_type == 'mynet':
        model = MyNet()
    elif cfg.model_type == 'resnet18':
        model = ResNet18()
    else:
        raise NameError('Unknown model type')

    model.to(device)

    ##### DATALOADER #####
    ##### TODO: check dataset.py #####
    original_train_loader = get_dataloader(os.path.join(dataset_dir, 'train'),
                                           batch_size=cfg.batch_size, split='train')
    # 從 loader 中取出 dataset 物件
    train_dataset = original_train_loader.dataset
    
    val_loader = get_dataloader(os.path.join(dataset_dir, 'val'),
                                batch_size=cfg.batch_size, split='val')

    ##### Semi-Supervised Learning 邏輯 #####
    if cfg.use_pseudo_labeling:
        unlabeled_loader = get_dataloader(os.path.join(dataset_dir, 'unlabel'), 
                                          batch_size=cfg.batch_size, split='test')
        
        print("Loading pretrained ResNet18 for Pseudo-labeling...")
        #teacher = ResNet18().to(device)
        teacher = MyNet().to(device)
        #teacher.load_state_dict(torch.load('checkpoint/resnet18_best.pth'))
        teacher.load_state_dict(torch.load('checkpoint/mynet_best.pth'))
        
        # 產生 Pseudo-dataset
        pseudo_dataset = generate_pseudo_labels(teacher, unlabeled_loader, device, threshold=0.95)
        
        if pseudo_dataset is not None:
            print("Merging original train set with pseudo-labeled data...")
            combined_dataset = ConcatDataset([train_dataset, pseudo_dataset])
            # 重新定義 train_loader
            train_loader = DataLoader(combined_dataset, 
                                      batch_size=cfg.batch_size, 
                                      shuffle=True, 
                                      num_workers=2, # 建議開一點 worker 加速
                                      pin_memory=True,
                                      drop_last=True)
        else:
            train_loader = original_train_loader
    else:
        train_loader = original_train_loader

    ##### LOSS & OPTIMIZER #####
    criterion = nn.CrossEntropyLoss()
    if cfg.use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                                    momentum=0.9, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=cfg.milestones,
                                                     gamma=0.1)
    
    ##### TRAINING & VALIDATION #####
    ##### TODO: check train() in this file #####
    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          logfile_dir=logfile_dir,
          model_save_dir=model_save_dir,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          device=device)
    
if __name__ == '__main__':
    main()
