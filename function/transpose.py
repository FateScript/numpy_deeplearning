#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from IPython import embed


class Transpose:

    def __init__(self, transpose_array):
        self.trans_array = np.array(transpose_array)
        self.inverse_array = self.trans_array.argsort()

    def forward(self, x):
        return x.transpose(self.trans_array)

    def backward(self, grad_input):
        return grad_input.transpose(self.inverse_array)


def test_Transpose(input, grad_input):
    f = Transpose((2, 0, 1))
    output = f.forward(input)
    grad_output = f.backward(grad_input)
    return output, grad_output


if __name__ == "__main__":
    x = np.arange(24).reshape((2, 3, 4))
    y = np.arange(24).reshape((2, 3, 4)).transpose((2, 0, 1))
    val, grad = test_Transpose(x, y)
    embed(header="main")
