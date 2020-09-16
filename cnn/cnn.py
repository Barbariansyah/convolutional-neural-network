import numpy as np
from layers import *

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

if __name__ == "__main__":
    cnn = MyCnn()
    inp = []
    for i in range(6):
        inp.append(np.random.randint(0, 5, size=(6,5)))
    print(inp)

    cnn.add(Pooling([3,2], 2, 'avg'))
    res = cnn.feed_forward(inp)

    print(res)