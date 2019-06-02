#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from IPython import embed


class Softmax:

    def forward(self, x):
        exp_vec = np.exp(x)
        self.output = exp_vec / exp_vec.sum()
        return self.output

    def backward(self, grad_input):
        '''
        if i == j,  gradient is e^i * (sum - e^i) / sum^2
        if i != j,  gradient is -(e^i * e^j) / sum^2
        '''
        softmax_grad = -np.dot(self.output.transpose(), self.output)
        add_grad = np.diag(self.output.reshape(-1))
        softmax_grad += add_grad
        return np.dot(grad_input, softmax_grad)


def test_Sigmoid(input, grad_input):
    f = Softmax()
    output = f.forward(input)
    grad_output = f.backward(grad_input)
    return output, grad_output


if __name__ == "__main__":
    x = np.arange(-5, 5).reshape((1, -1))
    y = np.arange(x.shape[-1])
    y = np.ones(x.shape)
    val, grad = test_Sigmoid(x, y)
    embed(header="main")
