import numpy as np
from typing import List

def relu(x: float) -> float:
    return x if x > 0 else 0

def softmax(inp: List[np.array]) -> list:
    x = inp[0]
    e_x = np.exp(x - np.max(x))
    res = e_x / e_x.sum()
    return [res]

def calculate_de_dnet_last_layer(out: List[np.array], target_class: int) -> List[np.array]:
    res = []
    for j, p in enumerate(out[0]):
        if j == target_class:
            res.append(np.array([-(1-p)]))
        else:
            res.append(np.array([p]))
    return res

def calculate_average_partial_error(batch_partial_error: List[List[List[np.array]]]) -> List[List[np.array]]:
    #List Data
    #List layers
    #List FMs
    #np.array FM
    data_count = len(batch_partial_error)
    average_partial_error = batch_partial_error[0]
    
    for data_index in range(1, len(batch_partial_error)):
        for layer_idx in range(len(batch_partial_error[data_index])):
            for fm_idx, fm in enumerate(batch_partial_error[data_index][layer_idx]):
                average_partial_error[layer_idx][fm_idx] += fm
    
    for layer_idx in range(len(average_partial_error)):
        for fm_idx in range(len(average_partial_error[layer_idx])):
            average_partial_error[layer_idx][fm_idx] /= data_count

    return average_partial_error