import numpy as np
from typing import List
from abc import ABC
from .common import relu, softmax
from math import ceil

class Layer(ABC):
    def call(self, inp: List[np.array]) -> List[np.array]:
        pass

    def calculate_output_shape(self, inp: List[tuple]):
        pass

    def backward_pass(self, input_layer: List[np.array], de_dnet: List[np.array]):
        pass

class Conv2D(Layer):
    def __init__(self, padding_size: int, filter_count: int, filter_shape: np.array, stride_size: int, input_shape: np.array = None):
        self.padding_size = padding_size
        self.filter_count = filter_count
        self.filter_shape = filter_shape
        self.stride_size = stride_size
        self.input_shape = input_shape

    def call(self, inp: List[np.array]) -> List[np.array]:
        fm_in_size_x = self.input_shape[0][0] if self.input_shape is not None else inp[0].shape[0] + 2 * self.padding_size
        fm_in_size_y = self.input_shape[0][1] if self.input_shape is not None else inp[0].shape[1] + 2 * self.padding_size
        filter_x = self.filter_shape[0]
        filter_y = self.filter_shape[1]
        stride = self.stride_size

        assert (fm_in_size_x - filter_x) % stride == 0
        assert (fm_in_size_y - filter_y) % stride == 0

        conv_res = self._convolution(inp)
        res = self._activation(conv_res)

        return res

    def init_weight(self, input_size):
        self.filters = []
        self.biases = []
        for _ in range(self.filter_count):
            self.filters.append(np.random.random(
                (self.filter_shape[0], self.filter_shape[1])))
            self.biases.append(np.random.random())

    def _convolution(self, inp: List[np.array]) -> List[np.array]:
        fm_in_size_x = self.input_shape[0][0] if self.input_shape is not None else inp[0].shape[0] + 2 * self.padding_size
        fm_in_size_y = self.input_shape[0][1] if self.input_shape is not None else inp[0].shape[1] + 2 * self.padding_size
        filter_x = self.filter_shape[0]
        filter_y = self.filter_shape[1]
        stride = self.stride_size
        fm_x = ((fm_in_size_x - filter_x) // stride) + 1
        fm_y = ((fm_in_size_y - filter_y) // stride) + 1

        res = []
        for f, b in zip(self.filters, self.biases):
            fm = np.array([[0] * fm_x] * fm_y)
            for fm_in in inp:
                fm_in_padded = np.pad(fm_in, (self.padding_size, ), constant_values=0)

                for i in range(0, fm_x):
                    for j in range(0, fm_y):
                        i_stride, j_stride = i * stride, j * stride
                        receptive_field = fm_in_padded[i_stride:i_stride+filter_x, j_stride:j_stride+filter_y]
                        value = np.sum(np.multiply(f, receptive_field))
                        fm[i, j] = fm[i, j] + value

            for i in range(fm_x):
                for j in range(fm_y):
                    fm[i, j] += b
                    
            res.append(fm)

        return res

    def _activation(self, conv_res: List[np.array]) -> List[np.array]:
        reluv = np.vectorize(relu)
        return [reluv(fm) for fm in conv_res]

    def calculate_output_shape(self, inp: List[tuple]):
        res = []

        for i in range(self.filter_count):
            row = ((inp[0][0] + 2 * self.padding_size -
                    self.filter_shape[0]) // self.stride_size) + 1
            column = ((inp[0][1] + 2 * self.padding_size -
                       self.filter_shape[1]) // self.stride_size) + 1
            res.append((row, column))

        return res

    def update_weights(self, partial_error: List[np.array], learning_rate: float, momentum: float, prev_delta_w: List[np.array], de_db: List[np.array]):
        pass

    def backward_pass(self, input_layer: List[np.array], de_dnet: List[np.array]):
        pass

class Pooling(Layer):
    def __init__(self, filter_shape: np.array, stride_size: int = 2, mode: str = 'max'):
        self.filter_shape = filter_shape
        self.stride_size = stride_size
        self.mode = mode

    def call(self, inp: List[np.array]) -> List[np.array]:
        res = []
        stride = self.stride_size
        filter_row = self.filter_shape[0]
        filter_column = self.filter_shape[1]
        reduced_map_size_row = (
            inp[0].shape[0] // stride) + (0 if inp[0].shape[0] % 2 == 0 else 1)
        reduced_map_size_column = (
            inp[0].shape[1] // stride) + (0 if inp[0].shape[1] % 2 == 0 else 1)
        for fm in inp:
            reduced_map = np.array(
                [[0.0] * reduced_map_size_column] * reduced_map_size_row)
            for i in range(0, fm.shape[0], stride):
                for j in range(0, fm.shape[1], stride):
                    red = np.amax(fm[i:i+filter_row, j:j+filter_column]) if self.mode == 'max' else np.mean(
                        fm[i:i+filter_row, j:j+filter_column])
                    reduced_map[i//stride, j//stride] = red
            res.append(reduced_map)
        return res

    def calculate_output_shape(self, inp: List[tuple]):
        res = []

        for i in range(len(inp)):
            row = ceil((inp[0][0] - self.filter_shape[0]) / self.stride_size) + 1
            column = ceil((inp[0][1] - self.filter_shape[1]) /
                      self.stride_size) + 1
            res.append((row, column))

        return res

    def backward_pass(self, input_layer: List[np.array], de_dnet: List[np.array]):
        return [], de_dnet

class Flatten(Layer):
    def __init__(self):
        pass

    def call(self, inp: List[np.array]) -> List[np.array]:
        res = []
        for i in inp:
            for j in i:
                res.append(j)

        res = np.array(res)
        res = res.flatten()
        return [res]

    def calculate_output_shape(self, inp: List[tuple]):
        res = 0

        for fm_shape in inp:
            res += fm_shape[0] * fm_shape[1]

        return [(res,)]

    def backward_pass(self, input_layer: List[np.array], de_dnet: List[np.array]):
        return [], de_dnet

class Dense(Layer):
    def __init__(self, unit_count: int, activation_function: str = 'relu'):
        self.unit_count = unit_count
        self.activation_function = activation_function

    def call(self, inp: List[np.array]) -> List[np.array]:
        result_dot_matrix = np.dot(inp[0], self.filters)
        result_dot_matrix = [np.add(result_dot_matrix, self.bias_weight)]
        result = self._activation(result_dot_matrix)

        return result

    def init_weight(self, input_size: List[tuple]):
        self.filters = np.random.random(
            (input_size[0][0], self.unit_count))
        self.bias_weight = np.random.random(
            self.unit_count)

    def _activation(self, conv_res: List[np.array]) -> List[np.array]:
        if self.activation_function == 'relu':
            reluv = np.vectorize(relu)
            result = [reluv(fm) for fm in conv_res]
        else:
            result = softmax(conv_res)
            
        return result

    def calculate_output_shape(self, inp: List[tuple]):
        return [(self.unit_count,)]

    def update_weights(self, partial_error: List[np.array], learning_rate: float, momentum: float, prev_delta_w: List[np.array], de_db: List[np.array]):
        pass

    def backward_pass(self, input_layer: List[np.array], de_dnet: List[np.array]):
        pass
