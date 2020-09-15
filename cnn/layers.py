import numpy as np
from typing import Union
from abc import ABC

class Layer(ABC):
    def call(self, inp: list) -> list:
        pass

class Conv2D(Layer):
    def __init__(self, input_shape: np.array, padding_size: int, filter_count: int, filter_shape: np.array, stride_size:int):
        return None
    
    def call(self, inp: list) -> list:
        return None

class Pooling(Layer):
    def __init__(self, filter_shape: np.array, stride_size: int =2 , mode: str ='max'):
        self.filter_shape = filter_shape
        self.stride_size = stride_size
        self.mode = mode

    def call(self, inp: list) -> list:
        res = []
        stride = self.stride_size
        filter_x = self.filter_shape[0]
        filter_y = self.filter_shape[1]
        reduced_map_size = (len(inp[0])//2) + (len(inp[0])%2)
        for fm in inp:
            reduced_map = np.array([[0.0] * reduced_map_size] * reduced_map_size)
            for i in range(0, len(fm), stride):
                for j in range(0, len(fm), stride):
                    red = np.amax(fm[i:i+filter_x, j:j+filter_y]) if self.mode=='max' else np.mean(fm[i:i+filter_x, j:j+filter_y])
                    reduced_map[i//2, j//2] = red
            res.append(reduced_map)
        return res

class Dense(Layer):
    def __init__(self, unit_count: int, activation_function : str = 'relu'):
        return None

    def call(self, inp: list) -> list:
        return None

if __name__ == "__main__":
    p = Pooling([2,2], 2, 'avg')
    inp = []
    for i in range(6):
        inp.append(np.random.randint(0, 5, size=(6,6)))
    print(inp)
    print(p.call(inp))