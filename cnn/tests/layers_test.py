import unittest
import numpy as np
from ..layers import Conv2D, Pooling, Flatten


class Conv2DTC(unittest.TestCase):
    def setUp(self):
        self.layer = Conv2D(0, 2, np.array([3, 3]), 1)
        self.layer.init_weight(np.array([3,3]))
        self.layer.biases = [0] * len(self.layer.biases)

    def test_call(self):
        self.layer.filters = [
            np.array([[-1, 1, -1],
                      [-1, 1, -1],
                      [-1, 1, -1]]),
            np.array([[0, 0, 0],
                      [1, 1, 1],
                      [0, 0, 0]])
        ]
        inp = [np.array([[244,  35, 227],
                         [178, 127, 222],
                         [172, 115, 188]])]
        res = self.layer.call(inp)
        self.assertTrue(np.array_equal(
            [np.array([[0]]), np.array([[527]])], res))

    def test_call_padding(self):
        self.layer.padding_size = 1
        self.layer.filters = [
            np.array([[-1, 1, -1],
                      [-1, 1, -1],
                      [-1, 1, -1]]),
        ]
        inp = [np.array([[244,  35, 227],
                         [178, 127, 222],
                         [172, 115, 188]])]
        res = self.layer.call(inp)
        self.assertTrue(np.array_equal(
            [np.array([[260, 0, 287], [317, 0, 360], [108, 0, 168]])], res))

    def test_call_stride3(self):
        self.layer.init_weight(np.array([6,6]))
        self.layer.biases = [0] * len(self.layer.biases)
        self.layer.padding_size = 0
        self.layer.stride_size = 3
        self.layer.filters = [
            np.array([[-1, 1, -1],
                      [-1, 1, -1],
                      [-1, 1, -1]]),
        ]
        inp = [np.array([[244,  35, 244,  35,  35, 244],
                         [178, 127, 178, 127, 127, 178],
                         [244,  35, 244,  35,  35, 244],
                         [178, 127, 178, 127, 127, 178],
                         [244,  35, 244,  35,  35, 244],
                         [178, 127, 178, 127, 127, 178]])]
        res = self.layer.call(inp)
        self.assertTrue(np.array_equal(
            [np.array([[0, 0], [0, 0]])], res))

class PoolingTC(unittest.TestCase):
    def setUp(self):
        self.layer = Pooling([2, 2], 2, 'avg')

    def test_call_4x4(self):
        inp = [np.array([[244,  35, 227,  57],
                         [178, 127, 222,  88],
                         [172, 115, 188, 150],
                         [0, 255,  11,  28]])]
        res = self.layer.call(inp)
        self.assertTrue(np.array_equal(
            [np.array([[146, 148.5], [135.5,  94.25]])], res))

    def test_call_2x2(self):
        inp = [np.array([[227,  57],
                         [222,  88]])]
        res = self.layer.call(inp)
        self.assertTrue(np.array_equal([np.array([[148.5]])], res))


class FlattenTC(unittest.TestCase):
    def setUp(self):
        self.layer = Flatten()

    def test_flatten(self):
        inp = [np.array([[244,  35, 227,  57],
                         [178, 127, 222,  88],
                         [172, 115, 188, 150],
                         [0, 255,  11,  28]])]
        res = self.layer.call(inp)
        self.assertTrue(16, len(res))
