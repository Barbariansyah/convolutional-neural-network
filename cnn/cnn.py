import numpy as np
from .layers import *
from .common import calculate_de_dnet_last_layer

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

    def fit(self, inp: list, epochs: int, batch_size: int = 5,learning_rate: float, momentum: float):
        for epoch in range(epochs):
            n_batch = math.ceil(len(inp) / batch_size)
            for n in range(n_batch):
                inp_batch = inp[ n * batch_size : (n + 1) * batch_size ]
                for data in inp_batch:
                    partial_error = self._back_propagation(data)
                # batch-size times:
                # 1. backpropagation
                # 2. save partial error
                
                # list of batch-size partial error
                
                # calculate average
                
                # update weight, consider last weight and momentum

                # save weight
        pass

    def _back_propagation(self, inp: List[np.array]):
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