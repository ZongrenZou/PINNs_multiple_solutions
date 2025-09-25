import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# r = np.random.uniform(size=[2000])
# theta = 2*np.pi * np.random.uniform(size=[2000])
# r = r.reshape([-1, 1])
# theta = theta.reshape([-1, 1])
# x = r * np.cos(theta)
# y = r * np.sin(theta)

# case 1: N = 3000
# case 2: N = 4000
# case 3: N = 5000
N = 3000

np.random.seed(8872)
x = 0 + 1 * np.random.uniform(size=[N])
y = 0 + 1 * np.random.uniform(size=[N])
x = x.reshape([-1, 1])
y = y.reshape([-1, 1])

x_b = np.concatenate(
    [np.linspace(0, 1, 101)[1:-1], np.linspace(0, 1, 101)[1:-1], np.zeros([101])[1:-1], np.ones([101])[1:-1]],
).reshape([-1, 1])
y_b = np.concatenate(
    [np.zeros([101])[1:-1], np.ones([101])[1:-1], np.linspace(0, 1, 101)[1:-1], np.linspace(0, 1, 101)[1:-1]],
).reshape([-1, 1])

print(x.shape, y.shape)
print(x_b.shape, y_b.shape)

plt.plot(x, y, ".")
plt.plot(x_b, y_b, "x")
plt.savefig("./points.png")
plt.show()


sio.savemat(
    "./data/data_case1.mat",
    {
        "xx": x, "yy": y, 
        "x_b": x_b, "y_b": y_b
    }
)