# -*- coding: utf-8 -*-

import glob
import os

import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from tools import combine_multilabel

class CustomDataset(Dataset):
    def __init__(self, root_dir, label_file, category, transform=None):
        self.images = glob.glob('{}/*jpg'.format(root_dir))
        print '[INFO] total images: {}'.format(len(self.images))
        self.root_dir = root_dir
        self.transform = transform
        # self.label_info = self.Normalize(label_file)
        if label_file:
            label_info = [json.loads(i) for i in open(label_file)][0]
            self.label_info = self.parse_label(label_info)
        else:
            self.label_info = {}
        self.category = category
        self.mean_std_df = pd.read_csv('./dataset/mean_std.csv')

    def parse_label(self, label_info):
        """
        parse the value of label_info
        """
        label_info_new = {}
        for key, value in label_info.items():
            label_info_new[key] = float(value)
        return label_info_new

    def Normalize(self, label_file):
        """
        normlize label value
        """
        label_info = [json.loads(i) for i in open(label_file)][0]
        max_value = max([float(value) for key, value in label_info.items()])
        min_value = min([float(value) for key, value in label_info.items()])
        label_info_new = {}
        for key, value in label_info.items():
            label_info_new[key] = (float(value) -min_value)/ (max_value - min_value)
        return label_info_new

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        img_name = os.path.basename(img_path)
        # sample_id, _, sub_point = img_name.split('.')[0].split('_')
        sample_id, sub_point = img_name.split('.')[0].split('_')
        index = '{}${}${}'.format(sample_id, sub_point, self.category)

        with Image.open(img_path) as img:
            image = img.convert('RGB')

        if self.transform:
            image = self.transform(image)

        value = self.label_info.get(index, np.array([10000]))

        if self.category in ['Elongation', 'Modulus', 'Yield strain', 'Yield stress', 'Failure stress']:
            self.mean = self.mean_std_df[self.category].iloc[1]
            self.std = self.mean_std_df[self.category].iloc[2]
            value = (value - self.mean) / self.std
        elif self.category in ['UTS']:
            pass
        else:
            raise Exception('ERROR in normlization')

        return image, value, index

    def reverse_transform(self, labels_normalized):
        '''Reverse normalization'''
        if self.category in ['Elongation', 'Modulus', 'Yield strain', 'Yield stress', 'Failure stress']:
            labels_orig = labels_normalized * self.std + self.mean
            # print('[Reverse transform] done')
        else:
            labels_orig = labels_normalized
            # print('[Reverse transform] do not need')
        return labels_orig


def split_indices(dataset, val_ratio=0.2, test_ratio=0.2, seed_val=None, seed_test=None):
    """
    split dataset and get indices
    """
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = range(dataset_size)

    test_num = int(np.floor(test_ratio * dataset_size))

    if seed_test:
        np.random.seed(seed_test)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[test_num:], indices[:test_num]
    
    indices_part = [i for i in indices if i not in test_indices]
    val_num = int(np.floor(val_ratio * len(indices_part)))
    if seed_val:
        np.random.seed(seed_val)
    np.random.shuffle(indices_part)
    train_indices, val_indices = indices_part[val_num:], indices_part[:val_num]

    print '[DATASET] train_indices[:5]', train_indices[:5]
    print '[DATASET] val_indices[:5]', val_indices[:5]
    print '[DATASET] test_indices[:5]', test_indices[:5]
    return train_indices, val_indices, test_indices


def split_indices_fix_test(dataset, test_index_list, val_ratio=0.16, seed_val=None, random_splt=False):
    """
    split dataset and get indices, but fix test dataset
    """
    # for 5-fold cv
    if seed_val not in range(5):
        raise Exception('ERROR in seed_val')

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = range(dataset_size)

    # select test dataset index
    test_indices = []
    for i in range(dataset_size):
        # if dataset[i][2] in test_index_list:
        if '$'.join(dataset[i][2].split('$')[:2]) in test_index_list:
            test_indices.append(i)
            # print dataset[i][2]

    train_val_indices = [i for i in range(dataset_size) if i not in test_indices]
    # check whether split train and val randomly
    if random_splt:
        val_num = int(np.floor(val_ratio * len(train_val_indices)))
        if seed_val:
            np.random.seed(seed_val)
        np.random.shuffle(train_val_indices)
        train_indices, val_indices = train_val_indices[val_num:], train_val_indices[:val_num]
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for idx, (train_index, val_index) in enumerate(kf.split(train_val_indices)):
            if idx == seed_val:
                train_indices = train_index
                val_indices = val_index
        np.random.seed(seed_val)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

    print '[DATASET] seed_val', seed_val
    print '[DATASET] train_indices[:5]', len(train_indices), train_indices[:5]
    print '[DATASET] val_indices[:5]', len(val_indices), val_indices[:5]
    print '[DATASET] test_indices[:5]', len(test_indices), test_indices[:5]
    return train_indices, val_indices, test_indices


def parser_batch(dataset, indices, batch_size=8):
    """
    parse dataset as sub_data_loader
    """
    sampler = SubsetRandomSampler(indices)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return data_loader

def save_test_index(test_loader, test_index_path):
    """
    save test index
    """
    test_indexs = ()
    for inputs, targets, index in test_loader:
        test_indexs += index
    test_indexs = list(test_indexs)
    with open(test_index_path, 'w') as f:
        for test_index in test_indexs:
            f.write(test_index+'\n')

def fetch_mean_std(data_loader):
    """
    calculate the mead and std of the dataset

    mean: [0.0105, 0.0369, 0.6360]
    std: [0.0432, 0.1013, 0.1116]
    """
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in data_loader:
        inputs = data[0]
        batch_samples = inputs.shape[0]
        inputs = inputs.view(batch_samples, inputs.shape[1], -1)
        mean += inputs.mean(2).sum(0)
        std += inputs.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    return mean, std

