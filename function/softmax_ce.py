#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from IPython import embed
from softmax import Softmax
from crossentropy import CrossEntropy


class Softmax_CE:

    def forward(self, x, target):
        exp_vec = np.exp(x)
        self.pred = exp_vec / exp_vec.sum()
        self.target = target
        return -(target * np.log(self.pred)).sum()

    def backward(self):
        return self.pred - self.target


def test_Softmax_CE(input, target):
    f = Softmax_CE()
    output = f.forward(input, target)
    grad_output = f.backward()

    softmax = Softmax()
    CE = CrossEntropy()
    pred = softmax.forward(input)
    div_output = CE.forward(pred, target)
    ce_grad = CE.backward()
    div_grad = softmax.backward(ce_grad)
    return output, grad_output, div_output, div_grad


if __name__ == "__main__":
    x = np.arange(-5, 5).reshape((1, -1))
    y = np.zeros(x.shape)
    y[0, 0] = 1
    val, grad, div_val, div_grad = test_Softmax_CE(x, y)
    embed(header="main")
