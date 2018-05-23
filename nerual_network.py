import torch
import torch.nn as nn

class pre_net(nn.Module):
    def __init__(self, input_size, output_size=[256,128]):
        super(prenet, self).__init__()
        input_size = [input_size] + output_size[:-1]
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) 
            for (input_dim, output_dim) in zip(input_size, output_size)])
        self.nonlinear_layer = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, input):
        for linear_layer in self.layers:
            input = self.dropout(self.nonlinear_layer(linear_layer(input)))
        return input

class batch_normal_conv1d(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, activation=None)
        super(batch_normal_conv1d).__init__()
        self.conv1d = 
