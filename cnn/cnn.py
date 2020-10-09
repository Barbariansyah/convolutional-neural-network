import numpy as np
from .layers import *
from .common import calculate_de_dnet_last_layer, calculate_average_partial_error
from typing import List
from math import ceil


class MyCnn(object):
    def __init__(self):
        self.layers = []
        self.layers_output_shape = []
        return None

    def add(self, layer: Layer):
        init_weight = getattr(layer, "init_weight", None)

        if len(self.layers):
            if callable(init_weight):
                layer.init_weight(self.layers_output_shape[-1])
            self.layers_output_shape.append(
                layer.calculate_output_shape(self.layers_output_shape[-1]))
        else:
            if callable(init_weight):
                layer.init_weight(layer.input_shape)
            self.layers_output_shape.append(
                layer.calculate_output_shape(layer.input_shape))

        self.layers.append(layer)
        return None

    def feed_forward(self, inp: list):
        temp = inp
        layers_input = []
        for layer in self.layers:
            layers_input.append(temp)
            temp = layer.call(temp)
        return temp, layers_input

    def fit(self, inp: list, target_classes: list, epochs: int, batch_size: int, learning_rate: float, momentum: float):
        # List[List[np.array]]
        prev_delta_w = []
        prev_delta_b = []
        self.layers[-1].last_layer = True

        for epoch in range(epochs):
            n_batch = ceil(len(inp) / batch_size)
            print(f'Epoch: {epoch}', end='', flush=True)

            for n in range(n_batch):
                print('.', end='', flush=True)
                batch_partial_error = []
                batch_partial_bias_error = []

                # back propagation for data in mini batch
                inp_batch = inp[n * batch_size: (n + 1) * batch_size]
                target_batch = target_classes[n *
                                              batch_size: (n + 1) * batch_size]
                for data, target in zip(inp_batch, target_batch):
                    partial_error, partial_bias_error = self._back_propagation(
                        data, target)
                    batch_partial_error.append(partial_error)
                    batch_partial_bias_error.append(partial_bias_error)

                # calculate avg partial error return List[List[np.array]]
                average_partial_error = calculate_average_partial_error(
                    batch_partial_error)
                average_partial_bias_error = calculate_average_partial_error(
                    batch_partial_bias_error)

                # update weight
                for i, layer in enumerate(self.layers):
                    update_weights = getattr(layer, "update_weights", None)
                    if callable(update_weights):
                        if len(prev_delta_w) > i:
                            prev_delta_w[i], prev_delta_b[i] = layer.update_weights(
                                average_partial_error[i], learning_rate, momentum, prev_delta_w[i], average_partial_bias_error[i], prev_delta_b[i])
                        else:
                            delta_w, delta_b = layer.update_weights(
                                average_partial_error[i], learning_rate, momentum, None, average_partial_bias_error[i], None)
                            prev_delta_w.append(delta_w)
                            prev_delta_b.append(delta_b)
                    else:
                        prev_delta_w.append([])
                        prev_delta_b.append([])

            print('', flush=True)

    def _back_propagation(self, inp: List[np.array], target_class: int) -> List[List[np.array]]:
        # 1. feed forward, save input
        # 2. backward pass, save error of layer-n+1

        result, layers_input = self.feed_forward(inp)
        last_layer_activation_layer = getattr(
            self.layers[-1], 'activation_function')
        de_dnet_list = [calculate_de_dnet_last_layer(
            result, target_class, last_layer_activation_layer)]
        de_dw_list = []
        de_db_list = []

        for i in range(len(self.layers)-1, -1, -1):
            layer_de_dw, layer_de_dnet, layer_de_db = self.layers[i].backward_pass(
                layers_input[i], de_dnet_list[0])
            de_dnet_list.insert(0, layer_de_dnet)
            de_dw_list.insert(0, layer_de_dw)
            de_db_list.insert(0, layer_de_db)

        return de_dw_list, de_db_list
