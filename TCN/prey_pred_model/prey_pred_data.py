import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch


def dfunc(y, t, a = 0.25, b = 0.12, c = 0.0025, d = 0.0013):
    # evaluate the right-hand-side at t
    f0 = a * y[0] - c * y[0] * y[1]
    f1 = - b * y[1] + d * y[0] * y[1]
    return [f0, f1]

def prey_pred_data(examples, seq_length, yinit = [80,50], t_range = (0.0, 200)):
    N = examples+seq_length+1  # Number of data points
    t = np.linspace(t_range[0], t_range[1], N)   # time grid
    ysol = odeint(dfunc, yinit, t)
    stack_data = np.zeros((examples, 2, seq_length+1))
    y0 = ysol[:, 0]
    y1 = ysol[:, 1]
    for i in range(examples):
        stack_data[i, 0, :] = y0[i:i+(seq_length + 1)]
        stack_data[i, 1, :] = y1[i:i+(seq_length + 1)]
    X = stack_data[:, :, :-1]
    y = stack_data[:, :, -1]
    return X, y

def prey_pred_data_randinit(examples, seq_length):
    batch_no = 100
    batch_samples = m.floor(examples / batch_no)
    full_data = np.zeros((examples, 2, seq_length+1))
    u_min = 50
    u_max = 100
    for i in range(batch_no):
        yinit = [u_min + (u_max-u_min)*torch.rand(1), u_min + (u_max-u_min)*torch.rand(1)]
        batch_data = prey_pred_data(batch_samples, seq_length, yinit = yinit)
        batch_data = np.concatenate( ( batch_data[0], batch_data[1].reshape(batch_samples, 2 , 1) ) , axis=2 )
        full_data[batch_samples*i:(batch_samples*(i+1))] = batch_data
    if examples % batch_no:
        batch_data = prey_pred_data(
            (examples % batch_no), seq_length, yinit = yinit)
        batch_data = np.concatenate( ( batch_data[0], batch_data[1].reshape(batch_samples, 2 , 1) ) , axis=2 )
        full_data[(batch_no*(i+1)):] = batch_data
    full_data = full_data[np.random.permutation(examples)]
    X = full_data[:, :, :-1]
    y = full_data[:, :, -1]
    return X, y

def prey_pred_data_init_withdt(examples, seq_length, yinit):
    N = 3000  # Number of data points
    if N < (examples+seq_length+1):
        print('No of examples should be smaller than 3000-seq_length-1')
        return
    tmin = 0.0     # starting t value
    tmax = 200.0   # final t value
    t = np.linspace(tmin, tmax, N)   # time grid
    ysol = odeint(dfunc, yinit, t)
    stack_data = torch.zeros(examples, 2, seq_length+2)
    y0 = torch.from_numpy(ysol[:, 0])
    y1 = torch.from_numpy(ysol[:, 1])
    for i in range(examples):
        stack_data[i, 0, :] = torch.cat(
            (torch.tensor([(tmax - tmin) / (N-1)]), y0[i:i+(seq_length + 1)].float()), 0)
        stack_data[i, 1, :] = torch.cat(
            (torch.tensor([(tmax - tmin) / (N-1)]), y1[i:i+(seq_length + 1)].float()), 0)
    # X = stack_data[:,:,:-1]
    # Y = stack_data[:,:,-1]
    return stack_data


def prey_pred_data_randinit_withdt(examples, seq_length):
    batch_no = 100
    batch_samples = m.floor(examples / batch_no)
    full_data = torch.zeros((examples, 2, seq_length+2))
    a = 50
    b = 100
    for i in range(batch_no):
        yinit = [a + (b-a)*torch.rand(1), a + (b-a)*torch.rand(1)]
        batch_data = prey_pred_data_init_withdt(
            batch_samples, seq_length, yinit)
        full_data[batch_samples*i:(batch_samples*(i+1))] = batch_data
    if examples % batch_no:
        yinit = [a + (b-a)*torch.rand(1), a + (b-a)*torch.rand(1)]
        batch_data = prey_pred_data_init_withdt(
            (examples % batch_no), seq_length, yinit)
        full_data[(batch_no*(i+1)):] = batch_data
    full_data = full_data[torch.randperm(examples)]
    X = full_data[:, :, :-1]
    Y = full_data[:, :, -1]
    return X, Y
