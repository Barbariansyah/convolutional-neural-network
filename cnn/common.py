import numpy as np
from typing import List

def relu(x: float) -> float:
    return x if x > 0 else 0

def softmax(inp: List[np.array]) -> list:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def calculate_de_dnet_last_layer(out: List[np.array]) -> List[np.array]:
    pass

def calculate_average_partial_error(batch_partial_error: List[List[List[np.array]]]) -> List[List[np.array]]:
    pass