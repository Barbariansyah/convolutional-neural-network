import numpy as np
from typing import List

def relu(x: float) -> float:
    return x if x > 0 else 0

def softmax(inp: List[np.array]) -> list:
    x = np.concatenate(inp, axis=0)
    e_x = np.exp(x - np.max(x))
    res = e_x / e_x.sum()
    res = [np.array([n]) for n in res]
    return res

def calculate_de_dnet_last_layer(out: List[np.array], target_class: int) -> List[np.array]:
    res = []
    for j, p in enumerate(out):
        if j == target_class:
            res.append(np.array([-(1-p[0])]))
        else:
            res.append(np.array([p[0]]))
    return res

def calculate_average_partial_error(batch_partial_error: List[List[List[np.array]]]) -> List[List[np.array]]:
    #List Data
    #List layers
    #List FMs
    #np.array FM
    pass