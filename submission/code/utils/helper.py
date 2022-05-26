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
<<<<<<< HEAD
            # data_blurred[i,j] =  cv2.morphologyEx(data_blurred[i,j,:,:], cv2.MORPH_CLOSE, kernel)
            data_blurred[i,j] =  cv2.dilate(data_blurred[i,j,:,:], kernel, iterations = 1)
=======
            data_blurred[i,j] =  cv2.morphologyEx(data_blurred[i,j,:,:], cv2.MORPH_OPEN, kernel)
>>>>>>> 52acb1ea8227143c22b69d99b3f8139209b43a68
    return data_blurred

def gauss(x, sigma):
    return np.exp(-(x**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

def gaussdxdx(x, sigma):
    return ((sigma**2 - x**2) / sigma**4) * gauss(x, sigma)

def filter_gaussdxdx(image, kernel_factor, sigma):
    faktor = kernel_factor * sigma * 2 + 1
    line = np.linspace(-kernel_factor * sigma, kernel_factor * sigma, faktor)
    filter = gaussdxdx(line, sigma=sigma)
    data_contrasted = ndimage.convolve(image, filter, mode="wrap")
    return data_contrasted
