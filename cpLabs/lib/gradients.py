import cv2
import numpy as np
from scipy import ndimage

SIMPLE_GRADIENT_OPERATOR_FLAG = 0
SIMPLE_GRADIENT_X_OPERATOR = np.array([[-1, 0, 1]])
SIMPLE_GRADIENT_Y_OPERATOR = np.array([[-1], [0], [1]])

SOBEL_OPERATOR_FLAG = 1
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

def compute_gradients(img, gradient_operator=0):
	if(len(img.shape) != 2):
		raise NotImplementedError, "compute_gradients implemented only for one-channel images"
	(height, width) = img.shape

	if gradient_operator == SOBEL_OPERATOR_FLAG:
		gradient_x_operator = SOBEL_X_OPERATOR
		gradient_y_operator = SOBEL_Y_OPERATOR
	else:
		gradient_x_operator = SIMPLE_GRADIENT_X_OPERATOR
		gradient_y_operator = SIMPLE_GRADIENT_Y_OPERATOR
	
	data = np.array(img, dtype=float)
	gradients_x = ndimage.convolve(data, gradient_x_operator)
	gradients_y = ndimage.convolve(data, gradient_y_operator)

	return np.absolute(gradients_x+gradients_y)
