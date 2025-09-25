import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


x = np.linspace(-1, 1, 201).reshape([-1, 1])
w = 6
u = np.sin(w*x) ** 3
u_x = 3 * np.sin(w*x) ** 2 * w * np.cos(w*x)
u_xx = 6 * np.sin(w*x) * w * w * np.cos(w*x) ** 2 - 3 * np.sin(w*x) ** 2 * w * np.sin(w*x) * w
kappa = 0.7
f = 0.01 * u_xx + kappa * np.tanh(u)

print(u)

sio.savemat(
    "./data/data_case_a.mat",
    {
        "x_train": x,
        "f_train": f,
        "x_test": x,
        "f_test": f,
        "u_test": u,
    }
)


x = np.linspace(-1, 1, 201).reshape([-1, 1])
w = 10
u = np.sin(w*x) ** 3
u_x = 3 * np.sin(w*x) ** 2 * w * np.cos(w*x)
u_xx = 6 * np.sin(w*x) * w * w * np.cos(w*x) ** 2 - 3 * np.sin(w*x) ** 2 * w * np.sin(w*x) * w
kappa = 0.7
f = 0.01 * u_xx + kappa * np.tanh(u)

print(u)

sio.savemat(
    "./data/data_case_b.mat",
    {
        "x_train": x,
        "f_train": f,
        "x_test": x,
        "f_test": f,
        "u_test": u,
    }
)
