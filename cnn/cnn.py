import numpy as np
from .layers import *


class MyCnn(object):
    def __init__(self):
        self.layers = []
        return None

    def add(self, layer: Layer):
        self.layers.append(layer)
        return None

    def feed_forward(self, inp: list):
        temp = inp
        for layer in self.layers:
            temp = layer.call(temp)
        return temp
