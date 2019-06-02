#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from IPython import embed


class Sigmoid:

    def forward(self, x):
        output = np.exp(x) / (np.exp(x) + 1)
        self.output = output
        return output

    def backward(self, grad_input):
        sigmoid_grad = self.output * (1 - self.output)
        return grad_input * sigmoid_grad


def test_Sigmoid(input, grad_input):
    f = Sigmoid()
    output = f.forward(input)
    grad_output = f.backward(grad_input)
    return output, grad_output


if __name__ == "__main__":
    x = np.arange(-5, 5).reshape((2, -1))
    y = np.ones(x.shape)
    val, grad = test_Sigmoid(x, y)
    embed(header="main")
