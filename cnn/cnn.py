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
        for layer in self.layers:
            temp = layer.call(temp)
        return temp
