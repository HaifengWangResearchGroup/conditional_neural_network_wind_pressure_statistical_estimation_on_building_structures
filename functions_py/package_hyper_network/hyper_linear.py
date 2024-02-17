"""
Author: Haifeng Wang

"""
import torch
from torch import nn
from torch.nn import init


class Linear_conditional(nn.Module):
    """
    Generate weights for linear layers
    """
    def __init__(
            self,
            hyper_input_size,
            linear_input_size,
            linear_output_size,
            dict_hyper_parameters=None):
        super().__init__()
        self.hyper_input_size = hyper_input_size
        self.linear_input_size = linear_input_size
        self.linear_output_size = linear_output_size
        int_output_size = (1 + linear_input_size) * linear_output_size

        self.list_expansion_factors = [0.1, 0.2, 0.4, 0.8]
        self.activation_1 = nn.ELU()

        if dict_hyper_parameters is None:
            tmp_int_output = int(
                int_output_size * self.list_expansion_factors[0])
            self.linear_1 = nn.Linear(
                in_features=hyper_input_size, out_features=tmp_int_output)
            self.normalization_1 = nn.LayerNorm(tmp_int_output)
            tmp_int_input = tmp_int_output
            tmp_int_output = int(
                int_output_size * self.list_expansion_factors[1])
            self.linear_2 = nn.Linear(
                in_features=tmp_int_input, out_features=tmp_int_output)
            self.normalization_2 = nn.LayerNorm(tmp_int_output)

            tmp_int_input = tmp_int_output
            tmp_int_output = int(
                int_output_size * self.list_expansion_factors[2])
            self.linear_3 = nn.Linear(
                in_features=tmp_int_input, out_features=tmp_int_output)
            self.normalization_3 = nn.LayerNorm(tmp_int_output)

            tmp_int_input = tmp_int_output
            tmp_int_output = int(
                int_output_size * self.list_expansion_factors[3])
            self.linear_4 = nn.Linear(
                in_features=tmp_int_input, out_features=tmp_int_output)
            self.normalization_4 = nn.LayerNorm(tmp_int_output)

            tmp_int_input = tmp_int_output
            self.linear_5 = nn.Linear(
                in_features=tmp_int_input,
                out_features=int_output_size)
        else:
            raise NotImplementedError
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.linear_1.weight.data)
        init.xavier_uniform_(self.linear_2.weight.data)
        init.xavier_uniform_(self.linear_3.weight.data)
        init.xavier_uniform_(self.linear_4.weight.data)
        init.xavier_uniform_(self.linear_5.weight.data)
        init.constant_(self.linear_1.bias.data, val=0)
        init.constant_(self.linear_2.bias.data, val=0)
        init.constant_(self.linear_3.bias.data, val=0)
        init.constant_(self.linear_4.bias.data, val=0)
        init.constant_(self.linear_5.bias.data, val=0)

    def forward(self, input_hyper):
        output = self.linear_1(input_hyper)
        output = self.activation_1(output)
        output = self.normalization_1(output)
        output = self.linear_2(output)
        output = self.activation_1(output)
        output = self.normalization_2(output)
        output = self.linear_3(output)
        output = self.activation_1(output)
        output = self.normalization_3(output)
        output = self.linear_4(output)
        output = self.normalization_4(output)
        output = self.activation_1(output)
        output = self.linear_5(output)
        output = output.view(
            input_hyper.shape[0],
            self.linear_output_size,
            self.linear_input_size + 1)
        return output


class Linear_no_train(nn.Module):
    def __init__(
            self, input_size, output_size,
            dtype=torch.float, bool_debug=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dtype = dtype
        self.bool_debug = bool_debug
        self.to(self.dtype)
        self.bool_debug = False

    def forward(self, x, linear_parameters):
        x = torch.unsqueeze(x, 1)
        weights_input, bias_input \
            = linear_parameters.split([self.input_size, 1], dim=2)
        output = torch.matmul(x, weights_input.transpose(1, 2)) \
            + bias_input.transpose(1, 2)
        return output.view(output.shape[0], output.shape[2])
