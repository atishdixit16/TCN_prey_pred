import math as m
import numpy as np
from scipy.integrate import odeint


def dfunc(y, t, a=0.25, b=0.12, c=0.0025, d=0.0013):
    # evaluate the right-hand-side at t
    f0 = a * y[0] - c * y[0] * y[1]
    f1 = - b * y[1] + d * y[0] * y[1]
    return [f0, f1]


def prey_pred_data(examples, seq_length, yinit=[80, 50], t_range=(0.0, 200), add_dt=False):
    N = examples+seq_length+1  # Number of data points
    t = np.linspace(t_range[0], t_range[1], N)   # time grid
    ysol = odeint(dfunc, yinit, t)
    stack_data = np.zeros((examples, 2, seq_length+1+add_dt))
    y0 = ysol[:, 0]
    y1 = ysol[:, 1]
    for i in range(examples):
        if add_dt:
            dt = (t_range[1] - t_range[0]) / (N-1)
            stack_data[i, 0, :] = np.concatenate(
                (np.array([dt]),  y0[i:i+(seq_length + 1)]))
            stack_data[i, 1, :] = np.concatenate(
                (np.array([dt]),  y1[i:i+(seq_length + 1)]))
        else:
            stack_data[i, 0, :] = y0[i:i+(seq_length + 1)]
            stack_data[i, 1, :] = y1[i:i+(seq_length + 1)]
    X = stack_data[:, :, :-1]
    y = stack_data[:, :, -1]
    return X, y


def prey_pred_data_randinit(examples, seq_length, yinit=[80, 50], t_range=(0.0, 200), add_dt=False):
    batch_no = 100
    batch_samples = m.floor(examples / batch_no)
    full_data = np.zeros((examples, 2, seq_length+1+add_dt))
    u_min = 50
    u_max = 100
    for i in range(batch_no):
        y0, y1 = u_min + (u_max-u_min)*np.random.rand(1), u_min + \
            (u_max-u_min)*np.random.rand(1)
        batch_data = prey_pred_data(batch_samples, seq_length, yinit=[
                                    y0.item(), y1.item()], add_dt=add_dt)
        batch_data = np.concatenate(
            (batch_data[0], batch_data[1].reshape(batch_samples, 2, 1)), axis=2)
        full_data[batch_samples*i:(batch_samples*(i+1))] = batch_data
    if examples % batch_no:
        batch_data = prey_pred_data(
            (examples % batch_no), seq_length, yinit=yinit, add_dt=add_dt)
        batch_data = np.concatenate(
            (batch_data[0], batch_data[1].reshape(batch_samples, 2, 1)), axis=2)
        full_data[(batch_no*(i+1)):] = batch_data
    full_data = full_data[np.random.permutation(examples)]
    X = full_data[:, :, :-1]
    y = full_data[:, :, -1]
    return X, y
