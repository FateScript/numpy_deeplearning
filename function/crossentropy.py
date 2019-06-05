#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from IPython import embed


class CrossEntropy:

    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return -(target * np.log(pred)).sum()

    def backward(self):
        return -self.target / self.pred


def test_CE(input, target):
    f = CrossEntropy()
    output = f.forward(input, target)
    grad_output = f.backward()
    return output, grad_output


if __name__ == "__main__":
    x = np.arange(-5, 5).reshape((1, -1))
    x = np.exp(x) / np.exp(x).sum()
    y = np.zeros(x.shape)
    y[0, 0] = 1
    val, grad = test_CE(x, y)
    embed(header="main")
