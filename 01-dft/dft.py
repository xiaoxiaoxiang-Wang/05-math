import math

import numpy as np
from scipy.fftpack import fft, ifft


def my_dct1(x):
    assert x.ndim == 1
    size = x.shape[0]
    res = np.zeros(size, dtype=complex)
    for i in range(size):
        for j in range(size):
            rel = x[j] * math.cos(2 * math.pi * i * j / size)
            ima = -x[j] * math.sin(2 * math.pi * i * j / size)
            res[i] += complex(rel, ima)
    return res


def my_dct2(x):
    assert x.ndim == 1
    size = x.shape[0]
    n = np.arange(size)
    k = n.reshape((size, 1))
    m = np.exp(-2j * math.pi * k * n/size)
    return np.dot(m, x)


def my_fft(x):
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


def my_ifft(x):
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
    fft1 = fft(x)
    ifft1 = ifft(fft1)
    print("fft1", fft1)
    print("ifft1", ifft1)

    my_dct1 = my_dct1(x)
    my_dct2 = my_dct2(x)
    my_fft1 = my_fft(x)
    my_ifft1 = my_ifft(my_fft1)
    print("my_dct1", my_dct1)
    print("my_dct2", my_dct2)
    print("my_fft1", my_fft1)
    print("my_ifft1", my_ifft1)
