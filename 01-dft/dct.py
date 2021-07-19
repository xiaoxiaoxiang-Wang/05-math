import math

import numpy as np
from scipy.fftpack import dct, idct

import matplotlib.pyplot as plt


# 离散余弦变换
def my_dct(x):
    assert x.ndim == 1
    size = x.shape[0]
    res = np.zeros(size)
    for i in range(size):
        for j in range(size):
            if i == 0:
                res[i] += math.sqrt(1.0 / size) * x[j] * math.cos((2 * j + 1) * math.pi * i / (2 * size))
            else:
                res[i] += math.sqrt(2.0 / size) * x[j] * math.cos((2 * j + 1) * math.pi * i / (2 * size))
    return res


def my_idct(x):
    assert x.ndim == 1
    size = x.shape[0]
    res = np.zeros(size)
    for i in range(size):
        for j in range(size):
            if j == 0:
                res[i] += math.sqrt(1.0 / size) * x[j] * math.cos((2 * i + 1) * math.pi * j / (2 * size))
            else:
                res[i] += math.sqrt(2.0 / size) * x[j] * math.cos((2 * i + 1) * math.pi * j / (2 * size))
    return res


if __name__ == '__main__':
    x = np.array([1, 2, 3, 5, 7])
    # x = np.random.uniform(low=0,high=2,size=1024)
    x = np.random.normal(loc=0,scale=10, size=102400)
    dct2 = dct(x, norm='ortho')
    idct2 = idct(dct2, norm='ortho')
    print("dct2", dct2)
    print("dct2 sum", sum(dct2))
    print("dct2 sum2", sum(dct2**2))
    print("idct2", idct2)

    x1 = range(102400)

    plt.plot(x1,dct2)
    plt.show()
    my_dct2 = my_dct(x)
    my_idct2 = my_idct(my_dct2)
    print("my_dct2", my_dct2)
    print("my_idct2", my_idct2)
