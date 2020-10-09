import numpy as np
from typing import List


def relu(x: float) -> float:
    return x if x > 0 else 0


def softmax(inp: List[np.array]) -> list:
    x = inp[0]
    e_x = np.exp(x - np.max(x))
    res = e_x / e_x.sum()
    return [res]


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-1*x))


def calculate_de_dnet_last_layer(out: List[np.array], target_class: int, activation_function: str) -> List[np.array]:
    res = []
    if activation_function == 'softmax':
        for j, p in enumerate(out[0]):
            if j == target_class:
                res.append(-(1-p))
            else:
                res.append(p)
    elif activation_function == 'relu':
        for _, p in enumerate(out[0]):
            if p > 0:
                res.append(p)
            else:
                res.append(0)
    elif activation_function == 'sigmoid':
        for _, p in enumerate(out[0]):
            res.append(p * (1-p))

    return [np.array(res)]


def calculate_average_partial_error(batch_partial_error: List[List[List[np.array]]]) -> List[List[np.array]]:
    # List Data
    # List layers
    # List FMs
    # np.array FM
    data_count = len(batch_partial_error)
    average_partial_error = batch_partial_error[0]

    for data_index in range(1, len(batch_partial_error)):
        for layer_idx in range(len(batch_partial_error[data_index])):
            for fm_idx, fm in enumerate(batch_partial_error[data_index][layer_idx]):
                average_partial_error[layer_idx][fm_idx] += fm

    for layer_idx in range(len(average_partial_error)):
        for fm_idx in range(len(average_partial_error[layer_idx])):
            average_partial_error[layer_idx][fm_idx] = average_partial_error[layer_idx][fm_idx] / data_count

    return average_partial_error
