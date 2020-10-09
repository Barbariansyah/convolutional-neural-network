import numpy as np
from typing import List, Tuple
from abc import ABC
from .common import relu, softmax
from math import ceil

__all__ = ['Layer', 'Conv2D', 'Pooling', 'Flatten', 'Dense']

class Layer(ABC):
    def call(self, inp: List[np.array]) -> List[np.array]:
        pass

    def calculate_output_shape(self, inp: List[tuple]):
        pass

    def backward_pass(self, input_layer: List[np.array], de_dnet: List[np.array]) -> Tuple[List[np.array], List[np.array], List[np.array]]:
        pass

class Conv2D(Layer):
    def __init__(self, padding_size: int, filter_count: int, filter_shape: np.array, stride_size: int, input_shape: np.array = None):
        self.padding_size = padding_size
        self.filter_count = filter_count
        self.filter_shape = filter_shape # [width, height]
        self.stride_size = stride_size
        self.input_shape = input_shape # [[width, height]]

    def call(self, inp: List[np.array]) -> List[np.array]:
        fm_in_size_x = self.input_shape[0][0] if self.input_shape is not None else inp[0].shape[1] + 2 * self.padding_size
        fm_in_size_y = self.input_shape[0][1] if self.input_shape is not None else inp[0].shape[0] + 2 * self.padding_size
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
            self.biases.append(0)
        self.biases = [np.array(self.biases)]

    def _convolution(self, inp: List[np.array]) -> List[np.array]:
        fm_in_size_x = self.input_shape[0][0] if self.input_shape is not None else inp[0].shape[1] + 2 * self.padding_size
        fm_in_size_y = self.input_shape[0][1] if self.input_shape is not None else inp[0].shape[0] + 2 * self.padding_size
        filter_x = self.filter_shape[0]
        filter_y = self.filter_shape[1]
        stride = self.stride_size
        fm_x = ((fm_in_size_x - filter_x) // stride) + 1
        fm_y = ((fm_in_size_y - filter_y) // stride) + 1

        res = []
        for f, b in zip(self.filters, self.biases[0]):
            fm = np.array([[0] * fm_x] * fm_y)
            for fm_in in inp:
                fm_in_padded = np.pad(fm_in, (self.padding_size, ), constant_values=0)

                for i in range(0, fm_x):
                    for j in range(0, fm_y):
                        i_stride, j_stride = i * stride, j * stride
                        receptive_field = fm_in_padded[j_stride:j_stride+filter_y, i_stride:i_stride+filter_x]
                        value = np.sum(np.multiply(f, receptive_field))
                        fm[j, i] = fm[j, i] + value

            for i in range(fm_x):
                for j in range(fm_y):
                    fm[j, i] += b
                    
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

    def update_weights(self, partial_error: List[np.array], learning_rate: float, momentum: float, prev_delta_w: List[np.array], de_db: List[np.array], prev_delta_b: List[np.array]):
        delta_w = []
        delta_b = []

        for i in range(len(self.filters)):
            prev_delta_w_i = prev_delta_w[i] if prev_delta_w is not None else 0
            delta_w_i = learning_rate * partial_error[i] + momentum * prev_delta_w_i
            delta_w.append(delta_w_i)
            self.filters[i] = np.subtract(self.filters[i], delta_w_i)

        prev_delta_b_i = prev_delta_b[0] if prev_delta_b is not None else 0
        delta_b_i = learning_rate * de_db[0] + momentum * prev_delta_b_i
        delta_b.append(delta_b_i)
        self.biases[0] = np.subtract(self.biases[0], delta_b_i)

        return delta_w, delta_b

    def backward_pass(self, input_layer: List[np.array], de_dnet: List[np.array]):
        input_layer_size_x = self.input_shape[0][0] if self.input_shape is not None else inp[0].shape[1] + 2 * self.padding_size
        input_layer_size_y = self.input_shape[0][1] if self.input_shape is not None else inp[0].shape[0] + 2 * self.padding_size
        filter_x = self.filter_shape[0]
        filter_y = self.filter_shape[1]
        output_layer = self.call(input_layer)
        
        # Calculate dE/db and dE/dx
        de_db = []
        de_dbefore_relu = []
        for fm, de_dnet_fm in zip(output_layer, de_dnet):
            fm_x, fm_y = fm.shape[1], fm.shape[0]
            de_dbefore_relu.append(np.copy(de_dnet_fm))
            for i in fm_x:
                for j in fm_y:
                    if fm[j, i] <= 0:
                        de_dbefore_relu[j, i] = 0
            de_db.append(np.sum(de_dbefore_relu[-1]))
        de_db = [np.array(de_db)]

        dx_x = de_dbefore_relu[0].shape[1]
        dx_y = de_dbefore_relu[0].shape[0]

        # Calculate dE/dw from valid convolution with dE/dx as filter
        stride = input_layer_size_x // dx_x
        de_dw = []
        for f in de_dbefore_relu:
            de_dw_fm = np.array([[0] * filter_x] * filter_y)
            for fm_in in input_layer:
                fm_in_padded = np.pad(fm_in, (self.padding_size, ), constant_values=0)

                for i in range(0, filter_x):
                    for j in range(0, filter_y):
                        i_stride, j_stride = i * stride, j * stride
                        receptive_field = fm_in_padded[j_stride:j_stride+dx_y, i_stride:i_stride+dx_x]
                        value = np.sum(np.multiply(f, receptive_field))
                        de_dw_fm[j, i] = de_dw_fm[j, i] + value
                    
            de_dw.append(de_dw_fm)
        
        # Calculate dE/dnet from full convolution
        full_padding_size = (((input_layer_size_x - 1) * self.stride_size) - dx_x + filter_x) // 2
        dx_padded = [np.pad(de_dbefore_relu, (full_padding_size, ), constant_values=0)]
        de_dnet_fm = np.array([[0] * input_layer_size_x] * input_layer_size_y)
        for f, dxp in zip(self.filters, dx_padded):
            for i in range(0, input_layer_size_x):
                for j in range(0, input_layer_size_y):
                    i_stride, j_stride = i * stride, j * stride
                    receptive_field = dxp[j_stride:j_stride+dx_y, i_stride:i_stride+dx_x]
                    value = np.sum(np.multiply(f, receptive_field))
                    de_dnet_fm[j, i] = de_dnet_fm[j, i] + value

        de_dnet = [de_dnet_fm] * len(input_layer)

        return de_dw, de_dnet, de_db 


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
        stride = self.stride_size
        filter_row = self.filter_shape[0]
        filter_column = self.filter_shape[1]
        x, y = input_layer[0].shape
        de_dnet_copy = np.copy(de_dnet)
        for idx, de in enumerate(de_dnet_copy):
            fm = input_layer[idx]
            x_de, y_de = de.shape
            for i in range(x_de):
                for j in range(y_de):
                    temp = fm[i*stride:i*stride+filter_row, j*stride:j*stride+filter_column]
                    if self.mode == 'max':
                        max_p, max_q = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
                        temp = np.zeros((filter_row, filter_column))
                        temp[max_p, max_q] = de[i][j]
                    else:
                        temp = np.full((filter_row, filter_column), de[i][j])
                    fm[i*stride:i*stride+filter_row, j*stride:j*stride+filter_column] = temp
        return [], de_dnet_copy, []

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
        i = len(input_layer)
        j, k = input_layer[0].shape
        res_de_dnet = de_dnet[0].reshape(i, j, k)
        res_de_dnet = [x for x in res_de_dnet]
        return [], res_de_dnet, []

class Dense(Layer):
    def __init__(self, unit_count: int, activation_function: str = 'relu'):
        self.unit_count = unit_count
        self.activation_function = activation_function
        self.last_layer = False

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
        delta_w = []
        delta_b = []

        prev_delta_w_i = prev_delta_w[i] if prev_delta_w is not None else 0
        delta_w_i = learning_rate * partial_error[i] + momentum * prev_delta_w_i
        delta_w.append(delta_w_i)
        self.filters = np.subtract(self.filters, delta_w_i)

        prev_delta_b_i = prev_delta_b[0] if prev_delta_b is not None else 0
        delta_b_i = learning_rate * de_db[0] + momentum * prev_delta_b_i
        delta_b.append(delta_b_i)
        self.bias_weight = np.subtract(self.bias_weight, delta_b_i)

        return delta_w, delta_b

    def backward_pass(self, input_layer: List[np.array], de_dnet: List[np.array]):
        de_dw = []
        de_db = []
        de_dnet = []
        if not self.last_layer: 
            output_layer = self.call(input_layer)

            # Calculate dE/db and dE/dx
            de_dx = []
            de_dx.append(np.copy(de_dnet))
            for i in range(de_dx[0].shape[0]):
                if output_layer[0][i] <= 0:
                    de_dx[0][i] = 0

            de_db = [np.copy(de_dx[0])]

            # Calculate dE/dw
            de_dw = [np.dot(input_layer[0].T, de_dx[0])]

            # Calculate dE/dnet
            de_dnet = [np.matmul(de_dx[0], self.filters)]
        else:
            # if last layer
            # Calculate dE/db
            de_db = [np.copy(de_dnet[0])]

            # Calculate dE/dw
            de_dw = [np.matmul(input_layer[0].T, de_dnet[0])]

            # Calculate dE/dnet
            de_dnet = [np.matmul(de_dnet[0], self.filters)]

        return de_dw, de_dnet, de_db
