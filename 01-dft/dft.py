import math

import numpy as np
from scipy.fftpack import fft, ifft

# 离散傅里叶变换，采样数 N，周期2pi 采样频率2pi/M
# 时域信号认为是以N为周期的,因为逆变换恢复的时候延展后是以N为周期的
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

def my_idct1(x):
    assert x.ndim == 1
    size = x.shape[0]
    res = np.zeros(size, dtype=complex)
    for i in range(size):
        for j in range(size):
            rel = x[j] * math.cos(2 * math.pi * i * j / size)
            ima = -x[j] * math.sin(2 * math.pi * i * j / size)
            res[i] += complex(rel, ima)
    return res

def my_idct2(x):
    assert x.ndim == 1
    size = x.shape[0]
    n = np.arange(size)
    k = n.reshape((size, 1))
    m = np.exp(2j * math.pi * k * n/size)
    return np.real(np.dot(m, x))/size

if __name__ == '__main__':
    x = np.array([1, 2, 3, 5, 7])
    x = np.random.normal(loc=0, scale=1, size=64)
    fft1 = fft(x)
    ifft1 = ifft(fft1)
    print("fft1", fft1)
    print("ifft1", ifft1)

    my_dct1 = my_dct1(x)
    my_dct2 = my_dct2(x)
    my_ifft2 = my_idct2(my_dct1)
    print("my_dct1", my_dct1)
    print("my_dct2", my_dct2)
    print("my_ifft2", my_ifft2)
