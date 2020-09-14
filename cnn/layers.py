import numpy as np
from typing import Union
from abc import ABC

class Layer(ABC):
    def call(self, input: Union[list, np.array]) -> Union[list, np.array]:
        pass

class Conv2D(Layer):
    def __init__(self, input_shape: np.array, padding_shape: np.array, filter_count: int, filter_shape: np.array, stride_shape: np.array):
        return None
    
    def call(self, input: Union[list, np.array]) -> Union[list, np.array]:
        return None

class Pooling(Layer):
    def __init__(self, filter_shape: np.array, stride_shape:np.array , mode='max'):
        return None

    def call(self, input: Union[list, np.array]) -> Union[list, np.array]:
        return None

class Dense(Layer):
    def __init__(self, unit_count: int, activation_function : string = 'relu'):
        return None

    def call(self, input: Union[list, np.array]) -> Union[list, np.array]:
        return None