#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from IPython import embed


class ReLU:

    def forward(self, x):
        self.grad_mask = x < 0
        return np.maximum(x, 0)

    def backward(self, grad_input):
        grad_input[self.grad_mask] = 0
        return grad_input


def test_ReLU(input, grad_input):
    f = ReLU()
    output = f.forward(input)
    grad_output = f.backward(grad_input)
    return output, grad_output


if __name__ == "__main__":
    x = np.arange(-5, 5).reshape((2, -1))
    y = np.arange(-5, 5).reshape((2, -1))
    val, grad = test_ReLU(x, y)
    embed(header="main")
