#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from IPython import embed


class Dropout:

    def __init__(self, input_dim, output_dim, p=0.5, bias=True):
        '''
        @param p: keep probility of neurals, if p=1, dropout is equal to fc
        '''
        self.weight = np.random.normal(size=(input_dim, output_dim))
        self.weight_grad = np.zeros(self.weight.shape)
        self.bias = np.zeros((1, output_dim)) if bias else None
        self.bias_grad = np.zeros(self.bias.shape) if bias else None
        self.prob = p

    def train_forward(self, x):
        '''
        During forward process in training, every neural is droped with probility p
        @param x: input
        return output: tensor after applying the forward(training) process
        '''
        self.x = x
        self.neural_mask = np.random.binomial(1, self.prob, size=(1, self.weight.shape[0]))
        self.drop_x = self.neural_mask * x
        output = np.dot(self.drop_x, self.weight)
        if self.bias is not None:
            output += self.bias
        return output

    def inference_forward(self, x):
        '''
        During forward process in inferencing, no neural is droped, but to get the same
        expectation, output must multiply probility p
        @param x: input
        return output: tensor after applying the forward(infercing) process
        '''
        output = self.prob * np.dot(x, self.weight)
        if self.bias is not None:
            output += self.bias
        return output

    def backward(self, grad_input):
        '''
        @param grad_input: grad of upper layer
        return : grad_input for lower layer
        '''
        self.weight_grad += np.dot(self.drop_x.transpose(), grad_input)
        if self.bias is not None:
            self.bias_grad += grad_input
        grad_output = np.dot(grad_input, self.weight.transpose())
        return grad_output * self.neural_mask

    def update_grad(self, lr=1e-2):
        self.weight += lr * self.weight_grad
        if self.bias is not None:
            self.bias += lr * self.bias_grad

    def set_zero_grad(self):
        self.weight_grad = np.zeros(self.weight_grad.shape)
        if self.bias is not None:
            self.bias_grad = np.zeros(self.bias_grad.shape)


def train_dropout(x, gt, threshold=0.1, with_bias=True, p=0.5):
    input_dim, output_dim = x.shape[-1], gt.shape[-1]
    drop_fc = Dropout(input_dim, output_dim, p, with_bias)
    output = drop_fc.train_forward(x)
    while abs((gt - output).sum()) > threshold:
        drop_fc.set_zero_grad()
        output = drop_fc.train_forward(x)
        print("output tensor:{}".format(output))
        grad_input = 2 * (gt - output)  # l2 loss
        drop_fc.backward(grad_input)
        drop_fc.update_grad()
    return drop_fc


if __name__ == "__main__":
    x = np.ones((1, 4))
    gt = np.ones((1, 3))
    drop_fc = train_dropout(x, gt, threshold=0.0001, p=0.5)
    print("pred:{}".format(drop_fc.inference_forward(x)))
    embed(header="main")
