import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch

def dfunc (y, t):
    a=0.25
    b=0.12
    c=0.0025
    d=0.0013
    # evaluate the right-hand-side at t
    f0 = a * y[0] - c * y[0] * y[1]
    f1 = - b * y[1] + d * y[0] * y[1]
    return [f0, f1]

def prey_pred_data(examples, seq_length):
    N = examples + seq_length + 1  # Number of data points
    tmin = 0.0     # starting t value
    tmax = 200.0   # final t value
    yinit = [47,150]      # initial value of x
    t  = np.linspace(tmin, tmax, N)   # time grid
    ysol = odeint(dfunc, yinit, t)
    stack_data  = torch.zeros(examples, 2, seq_length+1)
    y0 = torch.from_numpy(ysol[:, 0])
    y1 = torch.from_numpy(ysol[:, 1])
    for i in range(examples):
        stack_data[i,0,:] = y0[i:i+(seq_length + 1)]
        stack_data[i,1,:] = y1[i:i+(seq_length + 1)]
    X = stack_data[:,:,:-1]
    Y = stack_data[:,:,-1]
    return X,Y

def prey_pred_data_randinit(examples, seq_length):
    N = seq_length + 1
    tmin = 0.0
    tmax = seq_length
    t  = np.linspace(tmin, tmax, N)   # time grid
    stack_data = torch.zeros(examples, 2, seq_length+1)
    for i in range(examples):
        yinit = [5+150*torch.rand(1) , 5+150*torch.rand(1) ]
        ysol = odeint(dfunc, yinit, t)
        stack_data[i,0,:] = torch.from_numpy(ysol[:,0])
        stack_data[i,1,:] = torch.from_numpy(ysol[:,1])
    X = stack_data[:,:,:-1]
    Y = stack_data[:,:,-1]
    return X,Y

