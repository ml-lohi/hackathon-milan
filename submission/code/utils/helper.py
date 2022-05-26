import numpy as np
from utils import processing
from scipy import ndimage
from scipy.fftpack import fft
import cv2


def read_data(file):
    data = np.loadtxt(file, dtype=np.complex128)
    return data.reshape(data.shape[0], 5, 3, 64, 64)

def blur(data):
    """
    data shape (n_times, n_radars, (image))
    """
    data_blurred = data.copy()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_blurred[i,j] = cv2.GaussianBlur(data_blurred[i,j,:,:], ksize=(5,5), sigmaX=1)
    return data_blurred

def morphology(data):
    """
    data shape (n_times, n_radars, (image))
    """
    data_blurred = data.copy()
    kernel = np.ones((5,5),np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # data_blurred[i,j] =  cv2.morphologyEx(data_blurred[i,j,:,:], cv2.MORPH_CLOSE, kernel)
            data_blurred[i,j] =  cv2.dilate(data_blurred[i,j,:,:], kernel, iterations = 1)
    return data_blurred

def to_real(data):
    """
    data shape (n_times, n_radars, (image))
    """
    data_real = np.asarray(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_real[i,j] = np.abs(data)[i,j,:,:]
    return data_real.astype(float)