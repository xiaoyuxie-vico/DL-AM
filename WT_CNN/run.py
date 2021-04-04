# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Dec. 2020
'''

import os
import sys
import time
import yaml

from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

current_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.dirname(os.path.join('..', current_path))
sys.path.append(root_path)

from utils.config_parser import ConfigParser
from utils.model_zoo import ResNet18Regression
from utils.dataset import CustomDataset
from utils.dataset import parser_batch
from utils.dataset import split_indices, split_indices_fix_test
from utils.tools import check_dir


# config
config_path = os.path.join(current_path, 'config.yml')
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    is_cuda = True
else:
    is_cuda = False


def main():
    check_dir('./data')

    # 1: UTS, 2: Elongation, 3: Yield stress
    tag_num = sys.argv[1]
    config = ConfigParser('config.yml', tag_num)

    # transformer
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.0092, 0.0146, 0.5314],
            std=[0.0363, 0.0555, 0.0681]),
    ])

    # all dataset
    dataset_all = CustomDataset(config.dataset_dir, config.label_file, config.tag, transform=data_transform)

    loss_dict = {}
    for idx in range(config.random_split_num):
        cv_results = defaultdict(list)
        # for cv_num in range(5):
        for cv_num in [0]:
            time_st = time.time()
            # split dataset
            if config.fix_test_dataset:
                # fix test dataset
                train_indices, val_indices, test_indices = split_indices_fix_test(
                    dataset_all, test_index_list=config.test_index_list, seed_val=cv_num)
            else:
                # random split test dataset bsed on seed_test
                train_indices, val_indices, test_indices = split_indices(
                    dataset_all, seed_val=cv_num, seed_test=idx)

            train_loader = parser_batch(dataset_all, train_indices)
            val_loader = parser_batch(dataset_all, val_indices)
            test_loader = parser_batch(dataset_all, test_indices)
            total_loader = parser_batch(dataset_all, range(len(dataset_all)))

            # initial model
            net = ResNet18Regression().to(device)
            if config.fine_tune_model:
                print(f'[INFO] load fine_tune_model: {config.fine_tune_model}')
                checkpoints = torch.load(config.fine_tune_model)
                net.load_state_dict(checkpoints)
            if is_cuda:
                net = net.cuda()

            lr = config.LR
            optimizer = torch.optim.Adam(
                net.parameters(), lr=lr, weight_decay=0.0001)

            loss_func = nn.MSELoss()
            loss_list = []
            for epoch in range(config.EPOCH):
                net.train()
                for step, (inputs, targets, _) in enumerate(train_loader):
                    targets = torch.reshape(targets, (targets.shape[0], -1))
                    targets = targets.float()
                    if is_cuda:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    outputs = net(inputs)

                    loss = loss_func(outputs, targets)
                    optimizer.zero_grad() 
                    loss.backward()
                    optimizer.step()

                    loss_list.append(loss.item())
                    if step % 5 == 0:
                        print(f'\t [Loss] idx: {idx}, cv_num: {cv_num}, \
                            epoch: {epoch}, step: {step}, loss: {round(loss.item(), 4)}')

                # change lr
                if epoch % config.stepwise == 0:
                    for param_group in optimizer.param_groups:
                        lr = lr * 0.3
                        param_group['lr'] = lr


if __name__ == "__main__":
    main()
