import numpy as np
from operator import itemgetter
from cpLabs.lib.gradients import compute_gradients

SIMPLE_OPERATOR_FLAG = 0
SOBEL_OPERATOR_FLAG = 1

def all_in_focus(images, gradient_operator=1):
	(height, width) = images[0].shape

	gradients = [compute_gradients(img, gradient_operator) for img in images]
	contributions = [np.zeros((height, width), np.uint8) for img in images]

	output = np.zeros((height, width), np.uint8)
	for y in range(0, height):
		for x in range(0, width):
			argmax = __argmax([gradients[0][y,x], gradients[1][y,x], gradients[2][y,x]])
			output[y,x] = images[argmax][y,x]
			contributions[argmax][y,x] = 255

	return output, contributions

def __argmax(array):
	index, element = max(enumerate(array), key=itemgetter(1))
	return index;