import torch
from torch import nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, dropout=None):
        super(NN, self).__init__()
        self.n_hid = n_layer - 2
        self.in_layer = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        nn.init.kaiming_uniform_(self.in_layer.weight, nonlinearity="relu")
        self.hid_layer = []
        for i in range(self.n_hid):
            layer = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            self.hid_layer.append(layer)
        self.out_layer = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        # if clip is not None: nn.init.uniform_(self.out_layer.weight, a=0, b=clip)


    def forward(self, x, mode='normal'):
        h = torch.nn.functional.relu(self.in_layer(x))
        for i in range(self.n_hid):
            h = self.dropout(h) if self.dropout is not None else h
            h = torch.nn.functional.relu(self.hid_layer[i](h))
        if mode == 'normal':
            h = torch.nn.functional.sigmoid(self.out_layer(h))
            return h
        else:
            x_in = h
            x_out = self.out_layer(x_in)
            return x_in, x_out


class NormNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, dropout=None):
        super(NormNN, self).__init__()
        self.n_hid = n_layer - 2
        self.in_layer = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.in_layer.weight, nonlinearity="relu")
        self.hid_layer = []
        for i in range(self.n_hid):
            layer = nn.Linear(hidden_dim, hidden_dim)
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            self.hid_layer.append(layer)
        self.out_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x, mode = 'normal'):
        h = torch.nn.functional.relu(self.in_layer(x))
        for i in range(self.n_hid):
            h = self.dropout(h) if self.dropout is not None else h
            h = torch.nn.functional.relu(self.hid_layer[i](h))
        norm = torch.norm(h, p=1, dim=-1, keepdim=True).repeat(1, h.size(dim=-1)) + 1e-16
        h = torch.div(h, norm)
        if mode == 'normal':
            h = torch.nn.functional.sigmoid(self.out_layer(h))
            return h
        else:
            x_in = h
            x_out = self.out_layer(x_in)
            return x_in, x_out

    
class SimpleCNN(nn.Module):
    def __init__(self, n_channel, n_hidden, num_out):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channel, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=n_hidden, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=num_out)

    def forward(self, x):
        print(x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=2, padding=1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=3, stride=2, padding=1))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x