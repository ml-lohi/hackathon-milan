import numpy as np
from utils import processing
from scipy import ndimage
from scipy.fftpack import fft
import cv2


def read_data(file):
    data = np.loadtxt(file, dtype=np.complex128)
    return data.reshape(data.shape[0], 5, 3, 64, 64)


def blur(data, length, N_radar):
    data_blurred = data.copy()
    for i in range(length):
        for j in range(N_radar):
            data_blurred[i,j] = cv2.GaussianBlur(data_blurred[i,j,:,:], ksize=(5,5), sigmaX=1)
    return data_blurred

def to_real(data, length, N_radar):
    data_real = np.asarray(data)
    for i in range(length):
        for j in range(N_radar):
            data_real[i,j] = np.abs(data)[i,j,:,:]
    return data_real.astype(float)