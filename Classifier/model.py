import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# basic classifier model
class MLP(nn.Module):

    def __init__(self, input_size, hiddens, output_size, device=None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dim_list = [input_size, *hiddens, output_size]
        self.device = device

        if device != None:
            torch.cuda.set_device(device)

        self.layers = []
        for (dim1, dim2) in zip(self.dim_list[:-2], self.dim_list[1:-1]):
            if self.device != None:
                self.layers.append(nn.Linear(dim1, dim2).cuda())
                self.layers.append(nn.Tanh().cuda())
            else:
                self.layers.append(nn.Linear(dim1, dim2))
                self.layers.append(nn.Tanh())
        if self.device != None:
            self.layers.append(nn.Linear(self.dim_list[-2], self.dim_list[-1]).cuda())
        else:
            self.layers.append(nn.Linear(self.dim_list[-2], self.dim_list[-1]))

        self.models = nn.Sequential(*self.layers)

    def forward(self, input):

        assert (input.shape[1] == self.input_size)

        if self.device != None:
            torch.cuda.set_device(self.device)
            input = input.cuda()

        output = self.models(input)
        return output

    def to_cpu(self):
        self.device = None
        for model in self.models:
            model = model.cpu()

    def to_cuda(self, device):
        self.device = device
        torch.cuda.set_device(self.device)
        for model in self.models:
            model = model.cuda()



# Improved MLP Classifier Model
class ImprovedMLP(nn.Module):

    def __init__(self, input_size, hiddens, output_size, dropout_rate=0.5, device=None):
        super(ImprovedMLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dim_list = [input_size] + hiddens + [output_size]
        self.device = device
        self.dropout_rate = dropout_rate

        if device is not None:
            torch.cuda.set_device(device)

        self.layers = []
        for (dim1, dim2) in zip(self.dim_list[:-2], self.dim_list[1:-1]):
            self.layers.append(nn.Linear(dim1, dim2))
            self.layers.append(nn.Tanh())
            self.layers.append(nn.BatchNorm1d(dim2))  # Batch normalization to stabilize training
            self.layers.append(nn.Dropout(self.dropout_rate))  # Dropout for regularization
            
        self.layers.append(nn.Linear(self.dim_list[-2], self.dim_list[-1]))

        if self.device is not None:
            self.models = nn.Sequential(*[layer.cuda() for layer in self.layers])
        else:
            self.models = nn.Sequential(*self.layers)

    def forward(self, input):
        assert input.shape[1] == self.input_size, "Expected input of size {} but got {}".format(self.input_size, input.shape[1])

        if self.device is not None:
            torch.cuda.set_device(self.device)
            input = input.cuda()

        output = self.models(input)
        return output

    def to_cpu(self):
        self.device = None
        for model in self.models:
            model = model.cpu()

    def to_cuda(self, device):
        self.device = device
        torch.cuda.set_device(self.device)
        for model in self.models:
            model = model.cuda()

# # Example usage
# input_size = 100
# hiddens = [256, 128, 64]
# output_size = 10
# dropout_rate = 0.2
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# model = ImprovedMLP(input_size, hiddens, output_size, dropout_rate, device)

# # Example data
# batch_size = 32
# input_tensor = torch.randn(batch_size, input_size)

# # Forward pass
# output = model(input_tensor)

# print(output.shape)  # Expected output shape: [batch_size, output_size]


# LSTM based classifier model
class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device=None):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device

        if device is not None:
            torch.cuda.set_device(device)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        if self.device is not None:
            self.lstm = self.lstm.cuda()

        self.fc = nn.Linear(hidden_size, output_size)
        
        if self.device is not None:
            self.fc = self.fc.cuda()

    def forward(self, input):

        assert input.shape[-1] == self.input_size, "Expected input of last dimension size {} but got {}".format(self.input_size, input.shape[-1])

        if self.device is not None:
            torch.cuda.set_device(self.device)
            input = input.cuda()

        out, _ = self.lstm(input)
        out = self.fc(out[:, -1, :])  # Select the last time step's output

        return out

    def to_cpu(self):
        self.device = None
        self.lstm = self.lstm.cpu()
        self.fc = self.fc.cpu()

    def to_cuda(self, device):
        self.device = device
        torch.cuda.set_device(self.device)
        self.lstm = self.lstm.cuda()
        self.fc = self.fc.cuda()

# # Example usage
# input_size = 10
# hidden_size = 50
# output_size = 5
# num_layers = 2
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# model = LSTM(input_size, hidden_size, output_size, num_layers, device)

# # Example data
# batch_size = 32
# sequence_length = 15
# input_tensor = torch.randn(batch_size, sequence_length, input_size)

# # Forward pass
# output = model(input_tensor)

# print(output.shape)  # Expected output shape: [batch_size, output_size]