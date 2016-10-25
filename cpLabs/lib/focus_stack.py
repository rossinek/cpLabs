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

def all_in_focus_crossing(images, gradient_operator=1):
	(height, width) = images[0].shape

	gradients = [compute_gradients(img, gradient_operator) for img in images]
	contributions = [np.zeros((height, width), np.float32) for img in images]

	output = np.zeros((height, width), np.uint8)

	s = np.zeros((height, width), np.float32)
	for i in range(0, len(images)):
		s += gradients[i]

	for i in range(0, len(images)):
		contributions[i] = gradients[i]*255/s
		contributions[i][s==0] = 85.
		contributions[i] = contributions[i].astype(np.uint8)
		output += ((images[i]/255.0)*contributions[i]).astype(np.uint8)

	return output.astype(np.uint8), contributions

def __argmax(array):
	index, element = max(enumerate(array), key=itemgetter(1))
	return index;