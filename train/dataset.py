import sys
import os
import json
from glob import glob
import math
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import time


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_dirs, config_path="../config/dataset.json", is_trans=True, linear_cost=False):

        try:
            with open(config_path) as f:
                settings = json.load(f)
        except FileNotFoundError:
            print("ERR: config file not found", file=sys.stderr)
            sys.exit(1)
        try:
            data_types = settings["data_types"]
            collision_safety_limit = settings["collision_safety_limit"]
            bumpiness_limit = settings["bumpiness_limit"]
            traj_len = settings["data_len"]
            if(linear_cost):
                self.collision_upper_bound = settings["collision_upper_bound"]
                self.collision_lower_bound = settings["collision_lower_bound"]
        except KeyError:
            print("ERR: config file is not valid (check keywords)", file=sys.stderr)
            sys.exit(1)

        dataset_sizes = []
        _data_types = data_types.copy()
        for dataset_dir in dataset_dirs:
            data_len_list = []
            for data_type in data_types:
                data_len = len(glob(os.path.join(dataset_dir, data_type, "*")))
                if data_len == 0:
                    print(data_type)
                    _data_types.remove(data_type)
                    print(f"Notice: data type {data_type} is not used")
                else:
                    data_len_list.append(data_len)
            if data_len_list == []:
                print(f"ERR: all data may be empty", file=sys.stderr)
                sys.exit(1)
            dataset_sizes.append(min(data_len_list))

        self.dataset_dirs = dataset_dirs
        self.data_types = _data_types
        self.dataset_sizes = dataset_sizes
        self.collision_safety_limit = collision_safety_limit
        self.bumpiness_limit = bumpiness_limit
        self.traj_len = traj_len
        self.is_trans =  is_trans
        self.transform = nn.Sequential(
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        ).to(torch.device('cuda'))

        self.linear_cost = linear_cost
        #self.collision_upper_bound = collision_upper_bound
        #self.collision_lower_bound = collision_lower_bound

    def find_dataset_path(self, data_type, index):

        dataset_num = 0
        sum_datasizes = 0
        for dataset_size in self.dataset_sizes:
            if index - sum_datasizes >= dataset_size:
                dataset_num += 1
                sum_datasizes += dataset_size
            else:
                break

        return os.path.join(self.dataset_dirs[dataset_num], data_type, f"{index - sum_datasizes}.pt")

    def load_data(self, data_type, index):

        if data_type not in self.data_types:
            return None

        data_path = self.find_dataset_path(data_type, index)
        data = torch.load(data_path)
        if data is None:
            print(f"ERR: {data_type} could not read", file=sys.stderr)
            sys.exit(1)

        return data

    def check_if_collided(self, lidar_scans):

        if lidar_scans is None:
            return 0
        mins, _ = torch.min(lidar_scans, 1)
        is_collided = torch.lt(mins, self.collision_safety_limit).to(torch.float32)

        return is_collided

    def check_if_collided_linear(self, lidar_scans):

        if lidar_scans is None:
            return 0
        mins, _ = torch.min(lidar_scans, 1)
        mins = torch.clamp(mins, min=self.collision_lower_bound, max=self.collision_upper_bound)
        is_collided = (self.collision_upper_bound - mins) / (self.collision_upper_bound - self.collision_lower_bound)

        return is_collided

    def check_if_bumpy(self, imu_measurements):

        if imu_measurements is None:
            return 0
        maxs, _ = torch.max(torch.abs(imu_measurements), 1)
        is_bumpy = torch.gt(maxs, self.bumpiness_limit).to(torch.float32)

        return is_bumpy

    def generate_output_data(self, data_types, index):

        data = [None] * len(data_types)
        for i, data_type in enumerate(data_types):
            data[i] = self.load_data(data_type, index).squeeze()
            if(data_type == "obs" and self.is_trans):
                data[i] = self.transform(data[i].to(torch.device('cuda')))
            if(data_type == "lidar"):
                if(self.linear_cost):
                    data[i] = self.check_if_collided_linear(data[i])
                else:
                    data[i] = self.check_if_collided(data[i])
            if(data_type == "imu"):
                data[i] = self.check_if_bumpy(data[i])

        return data

    def __len__(self):

        return sum(self.dataset_sizes)

    def __getitem__(self, index):

        data_types = ("obs", "acs", "pos", "lidar", "imu")
        data = self.generate_output_data(data_types, index)

        return data

class FullDataset(BaseDataset):

    def __getitem__(self, index):

        data_types = ("obs", "acs", "pos", "lidar")
        data = self.generate_output_data(data_types, index)

        return data

class WithoutBumpinessDataset(BaseDataset):

    def __getitem__(self, index):

        data_types = ("obs", "acs", "pos", "lidar")
        data = self.generate_output_data(data_types, index)

        return data

class OnlyCollisionDataset(BaseDataset):

    def __getitem__(self, index):

        data_types = ("obs", "acs", "lidar")
        data = self.generate_output_data(data_types, index)

        return data
