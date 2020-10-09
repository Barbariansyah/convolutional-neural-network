import unittest
from ..common import relu


class ReluTC(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(10, relu(10))

    def test_negative(self):
        self.assertEqual(0, relu(-5))

    def test_zero(self):
        self.assertEqual(0, relu(0))
