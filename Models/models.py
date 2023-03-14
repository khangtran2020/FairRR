import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))
        return x

class NormNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NormNN, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x1 = torch.nn.functional.relu(self.layer_1(x))
        # bound norm to 1
        x2 = torch.nn.functional.relu(self.layer_2(x1))
        norm = torch.norm(x2, p=1, dim=-1, keepdim=True).repeat(1, x2.size(dim=-1)) + 1e-16
        x3 = torch.div(x2, norm)
        x4 = self.layer_3(x3)
        x5 = torch.nn.functional.sigmoid(x4)
        return x5

class NormLogit(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NormLogit, self).__init__()
        self.layer_1 = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        norm = torch.norm(x, p=1, dim=-1, keepdim=True).repeat(1, x.size(dim=-1)) + 1e-16
        x = torch.div(x, norm)
        x = self.layer_1(x)
        out = torch.nn.functional.sigmoid(x)
        return out

class Logit(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Logit, self).__init__()
        self.layer_1 = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = self.layer_1(x)
        out = torch.nn.functional.sigmoid(x)
        return out

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1296, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x