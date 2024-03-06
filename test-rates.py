import numpy as np


def ref_phi_func(x, y, t):
    # return np.cos(np.pi*x)*np.cos(np.pi*y)*np.exp(-t)
    return np.cos(t) * np.cos(np.pi * x) * np.cos(np.pi * y)


def t_ref_phi_func(x, y, t):
    # return np.cos(np.pi*x)*np.cos(np.pi*y)*np.exp(-t)
    return -np.sin(t) * np.cos(np.pi * x) * np.cos(np.pi * y)


def ref_theta_func(x, y, thetac, t):
    # return np.cos(np.pi*x)*np.cos(np.pi*y)*np.exp(-t)+thetac
    return np.sin(t) * np.sin(np.pi * x) * np.sin(np.pi * y) + thetac


def t_ref_theta_func(x, y, thetac, t):
    # return np.cos(np.pi*x)*np.cos(np.pi*y)*np.exp(-t)+thetac
    return np.cos(t) * np.sin(np.pi * x) * np.sin(np.pi * y)
