import numpy as np
import matplotlib.pyplot as plot


def f_x(k, b):
    def f(x):
        return k * x + b
    return f


def find_coefficients(x, y):
    x = np.column_stack((np.ones((x.size,), dtype=int), x))
    x_transposed = np.matrix.transpose(x)
    return np.linalg.pinv(x_transposed.dot(x)).dot(x_transposed).dot(y)

if __name__ == '__main__':
    data = np.loadtxt('task_1_capital.txt', delimiter='\t\t', skiprows=1)

    b, k = find_coefficients(data[:, 0], data[:, 1])
    f = f_x(k, b)
    x1 = 50000; x2 = 350000

    plot.plot([x1, x2], [f(x1), f(x2)])
    plot.scatter(data[:, 0], data[:, 1], color='r')
    plot.show()
