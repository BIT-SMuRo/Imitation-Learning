"""
* File Name : imitatoiondata.py
* Author    : ZEFFIRETTI, HESH
* College   : Beijing Institute of Technology
* E-Mail    : zeffiretti@bit.edu.cn, hiesh@mail.com
"""

from __future__ import print_function, division
import os
from abc import ABC

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import csv

# ignore warnings
import warnings

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode


class ImitationData(Dataset, ABC):
    """
    InteractionData
    """

    def __init__(self, file_path, demo_no=1, time_scale=100):
        """
        Args:
            file_path (string): file storing data (txt)
            demo_no (int): no of the demo rat data
        """
        #####################csv
        self.raw_data = []
        with open(file_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                self.raw_data.append(row)
        self.raw_data = np.array(self.raw_data, dtype=np.float32)
        ###################

        ######################npy
        # self.raw_data = np.load(file_path)
        # self.raw_data = np.array(self.raw_data, dtype=np.float32)
        ########################

        self.data = torch.from_numpy(self.raw_data)
        # show data shape
        # print("data shape: ", self.data.shape)
        # set demo index
        if demo_no == 1:
            # demo index: [0,8)
            self.demo_index_start = 0
        else:
            # demo index: [8,16)
            self.demo_index_start = 8
        # set time scale
        self.time_scale = time_scale

    def __len__(self):
        return len(self.data[:, 0]) - self.time_scale

    def __getitem__(self, idx):
        """
        return data with:
            demo_stack: data[idx:idx+time_scale,demo index]
            demo_y: data[idx+time_scale,demo index]
            policy_y: data[idx+time_scale, !demo index]
        """
        demo_stack = self.data[
            idx : idx + self.time_scale,
            self.demo_index_start : self.demo_index_start + 8,
        ]
        demo_y = self.data[
            idx + self.time_scale, self.demo_index_start : self.demo_index_start + 8
        ]
        policy_y = self.data[
            idx + self.time_scale,
            8 - self.demo_index_start : 16 - self.demo_index_start,
        ]
        return demo_stack, demo_y, policy_y

    def init_policy(self):
        """
        initial policy for robot
        :return: data[:time scale,!demo index]
        """
        policy_y = self.data[0 : self.time_scale]
        return policy_y.view(1, self.time_scale, -1)
