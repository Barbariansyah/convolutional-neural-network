import numpy as np
from typing import List
from abc import ABC
from .common import relu

class Layer(ABC):
    def call(self, inp: List[np.array]) -> List[np.array]:
        pass

class Conv2D(Layer):
    def __init__(self, padding_size: int, filter_count: int, filter_shape: np.array, stride_size:int):
        self.padding_size = padding_size
        self.filter_count = filter_count
        self.filter_shape = filter_shape
        self.stride_size = stride_size

        self._init_weight()
    
    def call(self, inp: List[np.array]) -> List[np.array]:
        fm_in_size_x = inp[0].shape[0] + 2 * self.padding_size
        fm_in_size_y = inp[0].shape[1] + 2 * self.padding_size
        filter_x = self.filter_shape[0]
        filter_y = self.filter_shape[1]
        stride = self.stride_size

        assert (fm_in_size_x - filter_x) % stride == 0
        assert (fm_in_size_y - filter_y) % stride == 0

        conv_res = self._convolution(inp)
        res = self._activation(conv_res)

        return res
    
    def _init_weight(self):
        self.filters = []
        for _ in range(self.filter_count):
            self.filters.append(np.random.random((self.filter_shape[0], self.filter_shape[1])))
    
    def _convolution(self, inp: List[np.array]) -> List[np.array]:
        fm_in_size_x = inp[0].shape[0] + 2 * self.padding_size
        fm_in_size_y = inp[0].shape[1] + 2 * self.padding_size
        filter_x = self.filter_shape[0]
        filter_y = self.filter_shape[1]
        stride = self.stride_size
        fm_x = (fm_in_size_x - filter_x + 1) // stride 
        fm_y = (fm_in_size_y - filter_y + 1) // stride

        res = []
        for f in self.filters:
            fm = np.array([[0] * fm_x] * fm_y)
            for fm_in in inp:
                for i in range(0, fm_x, stride):
                    for j in range(0, fm_y, stride):
                        receptive_field = fm_in[i:i+filter_x, j:j+filter_y]
                        value = np.sum(np.multiply(f, receptive_field))
                        fm[i, j] = fm[i, j] + value
            res.append(fm)
        
        return res

    def _activation(self, conv_res: List[np.array]) -> List[np.array]:
        reluv = np.vectorize(relu)
        return [reluv(fm) for fm in conv_res]


class Pooling(Layer):
    def __init__(self, filter_shape: np.array, stride_size: int =2 , mode: str ='max'):
        self.filter_shape = filter_shape
        self.stride_size = stride_size
        self.mode = mode

    def call(self, inp: List[np.array]) -> List[np.array]:
        res = []
        stride = self.stride_size
        filter_x = self.filter_shape[0]
        filter_y = self.filter_shape[1]
        reduced_map_size = (len(inp[0])//stride) + (1 if len(inp[0])%2 != 1 else 0 )
        for fm in inp:
            reduced_map = np.array([[0.0] * reduced_map_size] * reduced_map_size)
            for i in range(0, len(fm), stride):
                for j in range(0, len(fm), stride):
                    red = np.amax(fm[i:i+filter_x, j:j+filter_y]) if self.mode=='max' else np.mean(fm[i:i+filter_x, j:j+filter_y])
                    reduced_map[i//stride, j//stride] = red
            res.append(reduced_map)
        return res

class Dense(Layer):
    def __init__(self, unit_count: int, activation_function : str = 'relu'):
        return None

    def call(self, inp: List[np.array]) -> List[np.array]:
        return None