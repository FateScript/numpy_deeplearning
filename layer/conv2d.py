#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from IPython import embed


class Conv2d:

    def __init__(self, k_size, in_channel, out_channel, stride=1, padding=0, bias=True):
        self.weight = np.random.normal(size=(out_channel, in_channel, k_size, k_size))
        self.weight_grad = np.zeros(self.weight.shape)
        self.bias = np.zeros((1, out_channel)) if bias else None
        self.bias_grad = np.zeros(self.bias.shape) if bias else None

    def forward(self, x):
        '''
        padding logic
        im2col logic
        matrix product
        '''
        pass

    def backward(self, grad_input):
        pass

    def update_grad(self, lr=1e-2):
        self.weight += lr * self.weight_grad
        if self.bias is not None:
            self.bias += lr * self.bias_grad

    def set_zero_grad(self):
        self.weight_grad = np.zeros(self.weight_grad.shape)
        if self.bias is not None:
            self.bias_grad = np.zeros(self.bias_grad.shape)


def train_conv2d_layer(x, gt):
    pass


if __name__ == "__main__":
    embed(header="main")
