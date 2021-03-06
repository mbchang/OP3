import torch
import torch.utils.data
from rlkit.torch.pytorch_util import from_numpy
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
import numpy as np

class RefinementNetwork(nn.Module):
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes,
            lstm_size,
            lstm_input_size,
            added_fc_input_size=0,
            batch_norm_conv=False,
            batch_norm_fc=False,
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.batch_norm_conv = batch_norm_conv
        self.batch_norm_fc = batch_norm_fc
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        self.lstm = nn.LSTM(lstm_input_size, lstm_size, num_layers=1, batch_first=True)

        for out_channels, kernel_size, stride, padding in \
                zip(n_channels, kernel_sizes, strides, paddings):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

        # find output dim of conv_layers by trial and add normalization conv layers
        test_mat = torch.zeros(1, self.input_channels, self.input_width,
                               self.input_height)  # initially the model is on CPU (caller should then move it to GPU if
        for conv_layer in self.conv_layers:
            test_mat = conv_layer(test_mat)
            #self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))

        fc_input_size = int(np.prod(test_mat.shape))
        # used only for injecting input directly into fc layers
        fc_input_size += added_fc_input_size

        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)

            #norm_layer = nn.BatchNorm1d(hidden_size)
            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)

            self.fc_layers.append(fc_layer)
            #self.fc_norm_layers.append(norm_layer)
            fc_input_size = hidden_size

        self.last_fc = nn.Linear(lstm_size, output_size)
        #self.last_fc.weight.data.uniform_(-init_w, init_w)
        #self.last_fc.bias.data.uniform_(-init_w, init_w)
        self.last_fc2 = nn.Linear(lstm_size, output_size)

        xcoords = np.expand_dims(np.linspace(-1, 1, self.input_width), 0).repeat(self.input_height, 0)
        ycoords = np.repeat(np.linspace(-1, 1, self.input_height), self.input_width).reshape((self.input_height, self.input_width))

        self.coords = from_numpy(np.expand_dims(np.stack([xcoords, ycoords], 0), 0))



    def forward(self, input, hidden1, hidden2, extra_input=None):

        # need to reshape from batch of flattened images into (channsls, w, h)
        # import pdb; pdb.set_trace()
        # h = input.view(input.shape[0],
        #                 self.input_channels-2,
        #                 self.input_height,
        #                 self.input_width)
        h = input

        coords = self.coords.repeat(input.shape[0], 1, 1, 1)
        h = torch.cat([h, coords], 1)

        h = self.apply_forward(h, self.conv_layers, self.conv_norm_layers,
                               use_batch_norm=self.batch_norm_conv)
        # flatten channels for fc layers
        h = h.view(h.size(0), -1)

        # if extra_input is not None:
        #     h = torch.cat((h, extra_input), dim=1)

        output = self.apply_forward(h, self.fc_layers, self.fc_norm_layers,
                               use_batch_norm=self.batch_norm_fc)

        if extra_input is not None:
            output = torch.cat([output, extra_input], dim=1)

        if len(hidden1.shape) == 2:
            hidden1, hidden2 = hidden1.unsqueeze(0), hidden2.unsqueeze(0)
        self.lstm.flatten_parameters()
        output, hidden = self.lstm(output.unsqueeze(1), (hidden1, hidden2))
        #output1 = self.output_activation(self.last_fc(output.squeeze()))

        output1 = self.output_activation(self.last_fc(output.squeeze()))
        output2 = self.output_activation(self.last_fc2(output.squeeze()))
        return output1, output2, hidden[0], hidden[1]

    def initialize_hidden(self, bs):
        return (Variable(ptu.from_numpy(np.zeros((1, bs, self.lstm_size)))),
                Variable(ptu.from_numpy(np.zeros((1, bs, self.lstm_size)))))

    def apply_forward(self, input, hidden_layers, norm_layers,
                      use_batch_norm=False):
        h = input
        for layer in hidden_layers:
            h = layer(h)
            #if use_batch_norm:
            #    h = norm_layer(h)
            h = self.hidden_activation(h)
        return h

