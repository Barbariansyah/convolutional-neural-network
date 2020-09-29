import numpy as np
from .layers import *


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
        for n in range(epochs):
            continue            
            # data-size / batch-size times
            
                # batch-size times:
                # 1. backpropagation
                # 2. save partial error
                
                # list of batch-size partial error
                
                # calculate average
                
                # update weight, consider last weight and momentum

                # save weight
        pass

    def _back_propagation(self, learning_rate: float, momentum: float):
        # 1. feed forward, save input
        # 2. backward pass, save error of layer-n+1
        
        for i in range(len(self.layers)-1, -1, -1):
            #call update weight
            continue
        pass
