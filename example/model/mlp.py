import torch.nn as nn
import torch

class SimpleLoss(nn.Module):
    def __init__(self):
        super(SimpleLoss, self).__init__()

    def forward(self, inputs):
        return torch.mean(inputs)

class MLP(nn.Module):
    def __init__(self, num_layer, dim):
        super().__init__()
        self.num_layer = num_layer
        self.dim = dim
        self.net = []
        for i in range(num_layer):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True)))
        self.net.append(SimpleLoss())
        self.net = nn.ModuleList(self.net)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.net):
            x = layer(x)
        return x