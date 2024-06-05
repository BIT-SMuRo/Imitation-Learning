"""
* College   : Beijing Institute of Technology
"""

# import required packages
import numpy as np
import matplotlib.pyplot as plt
import torch


class BehaviorCost(torch.nn.Module):
    """
    BehaviorCost: torch model with 2 hidden layers
    input: x, a vector of size (time,joint)
    flatten layer: [time,joint] -> [time*joint]
    hidden layer 1: 200 relu units
    hidden layer 2: 50 relu units
    hiddem layer 3: 16 relu units
    output layer: 1 relu units
    output: y, a vector of size (joint)
    """

    def __init__(self, input, joint=5, hidden1=200, hidden2=50, hidden3=16):
        super(BehaviorCost, self).__init__()
        # normilize the input
        self.norm = torch.nn.LayerNorm(joint)
        # flatten layer: [b,input,joint] -> [b,input*joint]
        self.flatten = torch.nn.Flatten()
        # 1st hidden layer: 200 relu units, [b,input*16] -> [b,200]
        self.hidden1 = torch.nn.Linear(in_features=input * joint, out_features=hidden1)
        # 2nd hidden layer: 50 relu units, [b,200] -> [b,50]
        self.hidden2 = torch.nn.Linear(hidden1, hidden2)
        # 3rd hidden layer: 50 relu units, [b,50] -> [b,16]
        self.hidden3 = torch.nn.Linear(hidden2, hidden3)
        # output layer: 1 relu units, [b,16] -> [b,1]
        self.output = torch.nn.Linear(hidden3, 1)
        # # combine all units with torch.sequential
        self.model = torch.nn.Sequential(
            self.norm,
            self.flatten,
            self.hidden1,
            torch.nn.ReLU(),
            self.hidden2,
            torch.nn.ReLU(),
            self.hidden3,
            torch.nn.ReLU(),
            self.output,
        )

    # forward pass
    def forward(self, x):
        return self.model(x)


def behav_loss(output, target):
    return output - target
