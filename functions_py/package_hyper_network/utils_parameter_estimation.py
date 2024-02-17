"""
Abbreviations:
    MIMO: Multiple Input Single Output
    L[0-9]: layer_[0-9] , e.g, L1 = layer_1
    [0-9]D: [0-9] dense layer
"""
import torch
from torch import nn

from package_hyper_network.hyper_linear import Linear_conditional
from package_hyper_network.hyper_linear import Linear_no_train


class MLP_MIMO_ND(nn.Module):
    """
    MLP multiple input multiple output with N dense layers
    ----------------------------------------------------------------------------
    inputs:
        - hyper_input_size
            size of input hyper parameter
        - lstm_input_size
            number of lstm feature (cam be approximately treated as number of
            mode shapes accounted)
        - lstm_hidden_size
            number of hidden state of LSTM
        - model_input_size
            number of inputs (for a MDOF system represnting a building, this
            is the number of loading pints/ floors)
    """

    def __init__(
            self, hyper_input_size, model_input_size, model_output_size,
            dict_hyper_parameter, bool_debug=False):

        super().__init__()
        weight_generation_module = Linear_conditional
        self.bool_debug = bool_debug
        self.hyper_input_size = hyper_input_size
        self.model_input_size = model_input_size
        self.model_output_size = model_output_size

        self.FSB_hidden_size = model_input_size * 4
        self.activation = nn.LeakyReLU()

        # Load hyper parameters
        self.list_layer_neuron_numbers \
            = dict_hyper_parameter["list_layer_neuron_numbers"]
        self.int_number_used_dense_layer \
            = dict_hyper_parameter["int_number_used_dense_layer"]
        assert self.int_number_used_dense_layer <= len(
                    self.list_layer_neuron_numbers) + 1

        # set up layers
        list_normalization_layers = []
        for i_layer in range(self.int_number_used_dense_layer - 1):
            temp_int = self.list_layer_neuron_numbers[i_layer]
            list_normalization_layers.append(nn.LayerNorm(temp_int))

        list_conditional_layers = []
        list_functional_layers = []

        list_conditional_layers.append(
            weight_generation_module(
                hyper_input_size, model_input_size,
                self.list_layer_neuron_numbers[0]))

        list_functional_layers.append(
            Linear_no_train(
                model_input_size,
                self.list_layer_neuron_numbers[0])
        )

        i_layer = 0  # To avoid erros when int_number_used_dense_layer == 2
        for i_layer in range(1, self.int_number_used_dense_layer - 1):
            list_conditional_layers.append(
                weight_generation_module(
                    hyper_input_size,
                    self.list_layer_neuron_numbers[i_layer - 1],
                    self.list_layer_neuron_numbers[i_layer]))

            list_functional_layers.append(
                Linear_no_train(
                    self.list_layer_neuron_numbers[i_layer - 1],
                    self.list_layer_neuron_numbers[i_layer])
            )

        list_conditional_layers.append(
            weight_generation_module(
                hyper_input_size,
                self.list_layer_neuron_numbers[i_layer],
                model_output_size))

        list_functional_layers.append(
            Linear_no_train(
                self.list_layer_neuron_numbers[i_layer],
                model_output_size)
        )

        self.list_normalization_layers \
            = nn.ModuleList(list_normalization_layers)
        self.list_conditional_layers = nn.ModuleList(list_conditional_layers)
        self.list_functional_layers = nn.ModuleList(list_functional_layers)

    def forward(self, model_input, hyper_input):

        if self.bool_debug is True:
            print("debug: input_t shape:", model_input.shape)

        for i_layer in range(self.int_number_used_dense_layer - 1):
            if i_layer == 0:
                weights = self.list_conditional_layers[i_layer](hyper_input)
                output = self.list_functional_layers[i_layer](
                    model_input, weights)
                output = self.list_normalization_layers[i_layer](output)
            else:
                weights = self.list_conditional_layers[i_layer](hyper_input)
                output = self.list_functional_layers[i_layer](
                    output, weights)
                output = self.list_normalization_layers[i_layer](output)
            output = self.activation(output)
        i_layer = self.int_number_used_dense_layer - 1
        output = self.list_functional_layers[i_layer](
            output,
            self.list_conditional_layers[i_layer](hyper_input))

        if self.bool_debug is True:
            print("debug: output shape", output.shape)
        return output


class EnsembleModel_MLP_MIMO_ND(nn.Module):
    def __init__(
            self, num_models,
            hyper_input_size, model_input_size, model_output_size,
            dict_hyper_parameter=None, bool_debug=False):
        super(EnsembleModel_MLP_MIMO_ND, self).__init__()
        self.models = nn.ModuleList([
            MLP_MIMO_ND(
                hyper_input_size=hyper_input_size,
                model_input_size=model_input_size,
                model_output_size=model_output_size,
                dict_hyper_parameter=dict_hyper_parameter)
            for _ in range(num_models)])

    def forward(self, x, hyper_input):
        outputs = [model(x, hyper_input) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

