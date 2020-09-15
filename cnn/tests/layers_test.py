import unittest
import numpy as np
from ..layers import Conv2D, Pooling

class Conv2DTC(unittest.TestCase):
    def setUp(self):
        self.layer = Conv2D(0, 2, np.array([3,3]), 1)
    
    def test_filters_initialized(self):
        self.assertIsNotNone(self.layer.filters)
        self.assertEqual(2, len(self.layer.filters))
        self.assertEqual(3, self.layer.filters[0].shape[0])
        self.assertEqual(3, self.layer.filters[0].shape[1])
    
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
        self.assertTrue(np.array_equal([np.array([[0]]), np.array([[527]])], res))

class PoolingTC(unittest.TestCase):
    def setUp(self):
        self.layer = Pooling([2,2], 2, 'avg')

    def test_call(self):
        inp = [np.array([[244,  35, 227,  57],
                        [178, 127, 222,  88],
                        [172, 115, 188, 150],
                        [  0, 255,  11,  28]])]
        res = self.layer.call(inp)
        self.assertTrue(np.array_equal([np.array([[146, 148.5], [135.5,  94.25]])], res))