import math

import numpy as np
from scipy.fftpack import fft, ifft


def my_fft(x):
    assert x.ndim == 1
    size = x.shape[0]
    if size == 1:
        return x
    if size % 2 != 0:
        raise ValueError("size of x must be a power of 2")
    odd = my_fft(x[1::2])
    even = my_fft(x[0::2])
    coe = np.exp(-2j * math.pi * np.arange(size)/size)
    mid = int(size/2)
    a = even+coe[:mid]*odd
    b = even+coe[mid:]*odd
    return np.concatenate([a,b])

def my_ifft(x):
    assert x.ndim == 1
    size = x.shape[0]
    if size == 1:
        return x
    if size % 2 != 0:
        raise ValueError("size of x must be a power of 2")
    odd = my_ifft(x[1::2])
    even = my_ifft(x[0::2])
    coe = np.exp(2j * math.pi * np.arange(size)/size)
    mid = int(size/2)
    a = even+coe[:mid]*odd
    b = even+coe[mid:]*odd
    return np.concatenate([a,b])/2

if __name__ == '__main__':
    x = np.array([1, 2, 3, 5,6,9,10,11])
    fft1 = fft(x)
    ifft1 = ifft(fft1)
    print("fft1", fft1)
    print("ifft1", ifft1)

    my_fft1 = my_fft(x)
    my_ifft1 = my_ifft(my_fft1)
    print("my_fft1", my_fft1)
    print("my_ifft1", my_ifft1)
