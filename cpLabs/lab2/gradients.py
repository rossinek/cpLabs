import cv2
import numpy as np
from scipy import ndimage

GRADIENT_X_OPERATOR = np.array([[-1, 0, 1]])
GRADIENT_Y_OPERATOR = np.array([[-1], [0], [1]])

def compute_gradients(img):
	if(len(img.shape) != 2):
		raise NotImplementedError, "compute_gradients implemented only for one-channel images"
	(height, width) = img.shape
	
	data = np.array(img, dtype=float)
	gradients_x = ndimage.convolve(data, GRADIENT_X_OPERATOR)
	gradients_y = ndimage.convolve(data, GRADIENT_Y_OPERATOR)

	return np.absolute(gradients_x+gradients_y)