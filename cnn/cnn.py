import numpy as np
from .layers import *
from .common import calculate_de_dnet_last_layer, calculate_average_partial_error
from typing import List

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

    def fit(self, inp: list, epochs: int, batch_size: int,learning_rate: float, momentum: float):
        #List[List[np.array]]
        prev_delta_w = []
        for epoch in range(epochs):
            n_batch = math.ceil(len(inp) / batch_size)
            for n in range(n_batch):
                batch_partial_error = []
                inp_batch = inp[ n * batch_size : (n + 1) * batch_size ]
                for data in inp_batch:
                    partial_error = self._back_propagation(data)
                    batch_partial_error.append(partial_error)
                
                #calculate avg partial error return List[List[np.array]]
                average_partial_error = calculate_average_partial_error(batch_partial_error)

                for i, layer in enumerate(self.layers):
                    update_weight = getattr(layer, "update_weight", None)
                    if callable(update_weight):
                        if prev_delta_w:
                            prev_delta_w[i] = layer.update_weight(average_partial_error[i], learning_rate, momentum, prev_delta_w[i])
                        else:
                            prev_delta_w.append(layer.update_weight(average_partial_error[i], learning_rate, momentum, None))

    def _back_propagation(self, inp: List[np.array]) -> List[List[np.array]]:
        # 1. feed forward, save input
        # 2. backward pass, save error of layer-n+1
        
        result, layers_input = self.feed_forward(inp)
        de_dnet_list = [calculate_de_dnet_last_layer(result)]
        de_dw_list = []

        for i in range(len(self.layers)-1, -1, -1):
            layer_de_dw, layer_de_dnet = layers[i].backward_pass(layers_input[i], de_dnet_list[0])
            de_dnet_list.insert(0, layer_de_dnet) 
            de_dw_list.insert(0, layer_de_dw)

        return de_dw_list