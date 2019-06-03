#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from IPython import embed


def reshaped(tensor):
    return tensor.reshape((1, -1, 1, 1))


class BatchNormalization:

    def __init__(self, channel_dim, momentum=0.1, eps=1e-5):
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros(channel_dim)
        self.running_var = np.ones(channel_dim)
        self.gamma = np.random.uniform(size=channel_dim)
        self.beta = np.zeros(channel_dim)
        self.gamma_grad = np.zeros(channel_dim)
        self.beta_grad = np.zeros(channel_dim)

    def train_forward(self, x):
        '''
        @param x: an tensor with (N, C, H, W) shape
        compute output with batch_mean/var, update mean/var with moving mean
        '''
        assert len(x.shape) == 4
        batch_mean = x.mean(axis=(0, 2, 3))
        batch_var = x.var(axis=(0, 2, 3))
        self.x_minus_mu = x - reshaped(batch_mean)
        self.norm_val = self.x_minus_mu / reshaped(np.sqrt(batch_var + self.eps))
        output = self.norm_val * reshaped(self.gamma) + reshaped(self.beta)

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        self.batch_mean = batch_mean
        self.batch_var = batch_var
        return output

    def inference_forward(self, x):
        '''
        @param x: an tensor with (N, C, H, W) shape
        compute output with running_mean/var fixed
        '''
        assert len(x.shape) == 4
        x_minus_mu = x - reshaped(self.running_mean)
        norm_val = x_minus_mu / reshaped(np.sqrt(self.running_var + self.eps))
        output = norm_val * reshaped(self.gamma) + reshaped(self.beta)
        return output

    def backward(self, grad_input):
        assert len(grad_input.shape) == 4
        # gamma and beta applys broadcast, so we use sum during backward process
        self.beta_grad = grad_input.sum(axis=(0, 2, 3))
        self.gamma_grad = (grad_input * self.norm_val).sum(axis=(0, 2, 3))
        loss_over_norm_grad = grad_input * reshaped(self.gamma)  # (N, C, H, W) shape
        norm_over_var_grad = (self.x_minus_mu * (reshaped(self.batch_var + self.eps))**(-1.5)
                              ).sum(axis=(0, 2, 3)) * -0.5  # (C) shape
        # shape of loss_over_var_grad is (C)
        loss_over_var_gard = loss_over_norm_grad.sum(axis=(0, 2, 3)) * norm_over_var_grad
        loss_over_mean_grad1 = (loss_over_norm_grad / reshaped(np.sqrt(self.batch_var + self.eps))
                                ).sum(axis=(0, 2, 3))  # (C) shape
        # shape of loss_over_mean_grad2 is (C)
        loss_over_mean_grad2 = loss_over_var_gard * (-2 * self.x_minus_mu).mean(axis=(0, 2, 3))
        loss_over_mean_grad = loss_over_mean_grad1 + loss_over_mean_grad2  # (C) shape
        loss_over_x1 = loss_over_norm_grad / reshaped(np.sqrt(self.batch_var + self.eps))  # (NCHW)
        loss_over_x2 = (loss_over_var_gard * 2 * self.x_minus_mu + loss_over_mean_grad
                        ) / grad_input.shape[0]  # (C) shape
        grad_output = loss_over_x1 + reshaped(loss_over_x2)
        return grad_output

    def update_grad(self, lr=1e-2):
        self.gamma += lr * self.gamma_grad
        self.beta += lr * self.beta_grad

    def set_zero_grad(self, ):
        self.gamma_grad = np.zeros(self.gamma_grad.shape)
        self.beta_grad = np.zeros(self.beta_grad.shape)


def train_conv2d_layer():
    pass


if __name__ == "__main__":
    x = np.random.normal(size=(2, 3, 8, 8))
    x = np.ones((2, 3, 8, 8))
    bn = BatchNormalization(channel_dim=3)
    y = bn.train_forward(x)
    embed(header="main")
