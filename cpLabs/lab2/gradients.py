import cv2
import numpy as np
from scipy import ndimage

SIMPLE_GRADIENT_X_OPERATOR = np.array([[-1, 0, 1]])
SIMPLE_GRADIENT_Y_OPERATOR = np.array([[-1], [0], [1]])

SOBEL_X_OPERATOR = np.array([
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1],
	])

SOBEL_Y_OPERATOR = np.array([
	[-1,-2, -1],
	[0,  0,  0],
	[1,  2,  1],
	])

def compute_gradients(img):
	if(len(img.shape) != 2):
		raise NotImplementedError, "compute_gradients implemented only for one-channel images"
	(height, width) = img.shape
	
	data = np.array(img, dtype=float)
	gradients_x = ndimage.convolve(data, SIMPLE_GRADIENT_X_OPERATOR)
	gradients_y = ndimage.convolve(data, SIMPLE_GRADIENT_Y_OPERATOR)

	return np.absolute(gradients_x+gradients_y)

def compute_sobel_gradients(img):
	if(len(img.shape) != 2):
		raise NotImplementedError, "compute_gradients implemented only for one-channel images"
	(height, width) = img.shape
	
	data = np.array(img, dtype=float)
	gradients_x = ndimage.convolve(data, SOBEL_X_OPERATOR)
	gradients_y = ndimage.convolve(data, SOBEL_Y_OPERATOR)

	return np.absolute(gradients_x+gradients_y)