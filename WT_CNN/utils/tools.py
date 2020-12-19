# -*- coding: utf-8 -*-

import os

import torch


def check_path(file_path):
    """
    Check whether the file_path exists, if not create it.
    """
    file_dir, file_name = os.path.split(file_path)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
        return True
    return False

def check_dir(dir_path):
    """
    Check whether the dir_path exists, if not create it.
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        return True
    return False

def normlize(label_info):
    """
    normlize a dict
    """
    max_value = max([float(value) for key, value in label_info.items()])
    min_value = min([float(value) for key, value in label_info.items()])
    label_info_new = {}
    for key, value in label_info.items():
        label_info_new[key] = (float(value) - min_value) / (max_value - min_value)
    return label_info_new

def targets_convert(targets):
    targets = targets.numpy()
    targets = targets.reshape(targets.shape[0], 1)
    targets = torch.tensor(targets)
    targets = targets.float()
    return targets

