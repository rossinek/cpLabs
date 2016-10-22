import cv2
from operator import itemgetter

from cpLabs.lab2.gradients import compute_gradients

def main():
	print "Task 1: Focal Stack"
	img1 = cv2.imread("cpLabs/static/images/focalstack/stack1.png", 0)
	img2 = cv2.imread("cpLabs/static/images/focalstack/stack2.png", 0)
	img3 = cv2.imread("cpLabs/static/images/focalstack/stack3.png", 0)

	images = [img1,img2,img3]
	gradients = [compute_gradients(img1), compute_gradients(img2), compute_gradients(img3)]
	output = img1.copy()
	(height, width) = img1.shape
	for y in range(0, height):
		for x in range(0, width):
			argmax = __argmax([gradients[0][y,x], gradients[1][y,x], gradients[2][y,x]])
			output[y,x] = images[argmax][y,x]

	cv2.imshow('Image',output)
	cv2.imshow('img1',img1)
	cv2.imshow('img2',img2)
	cv2.imshow('img3',img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def __argmax(array):
	index, element = max(enumerate(array), key=itemgetter(1))
	return index;

main()