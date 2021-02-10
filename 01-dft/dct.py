import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import dct, idct,fft

def my_dct(a):
    # a'[m] = a[m]       0<=m<=N-1
    # a'[m] = a[-m-1]    -N<=m<=-1
    assert a.ndim == 1
    







if __name__=='__main__':
    a = np.array([1,2])
    print(a.ndim)
    a_dct = dct(a,norm='ortho')
    a_dct2 = dct(a)
    print(a_dct)
    print(a_dct2)
    print(a_dct2/a_dct)