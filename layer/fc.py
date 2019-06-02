#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from IPython import embed


class FullyConnected:

    def __init__(self, input_dim, output_dim, bias=True):
        self.weight = np.random.normal(size=(input_dim, output_dim))
        self.weight_grad = np.zeros(self.weight.shape)
        self.bias = np.zeros((1, output_dim)) if bias else None
        self.bias_grad = np.zeros(self.bias.shape) if bias else None

    def forward(self, x):
        '''
        @param x: input
        return output: tensor after applying the forward process
        '''
        self.x = x  # save x for backward
        output = np.dot(x, self.weight)
        if self.bias is not None:
            output += self.bias
        return output

    def backward(self, grad_input):
        '''
        @param grad_input: grad of upper layer
        return : grad_input for lower layer
        '''
        self.weight_grad += np.dot(self.x.transpose(), grad_input)
        if self.bias is not None:
            self.bias_grad += grad_input
        grad_output = np.dot(grad_input, self.weight.transpose())
        return grad_output

    def update_grad(self, lr=1e-2):
        self.weight += lr * self.weight_grad
        if self.bias is not None:
            self.bias += lr * self.bias_grad

    def set_zero_grad(self):
        self.weight_grad = np.zeros(self.weight_grad.shape)
        if self.bias is not None:
            self.bias_grad = np.zeros(self.bias_grad.shape)


def train_fc(x, gt, threshold=0.1, with_bias=True):
    input_dim, output_dim = x.shape[-1], gt.shape[-1]
    fc = FullyConnected(input_dim, output_dim, with_bias)
    output = fc.forward(x)
    while abs((gt - output).sum()) > threshold:
        fc.set_zero_grad()
        output = fc.forward(x)
        print("output tensor:{}".format(output))
        grad_input = 2 * (gt - output)  # l2 loss
        fc.backward(grad_input)
        fc.update_grad()
    return fc


if __name__ == "__main__":
    x = np.ones((1, 4))
    gt = np.ones((1, 3))
    fc = train_fc(x, gt, 0.0001)
    print("pred:{}".format(fc.forward(x)))
    embed(header="main")
