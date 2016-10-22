import cv2
import numpy as np
from operator import itemgetter

from cpLabs.lab2.gradients import compute_gradients, compute_sobel_gradients

def main():
	print "Task 1: Focal Stack"
	img1 = cv2.imread("cpLabs/static/images/focalstack/stack1.png", 0)
	img2 = cv2.imread("cpLabs/static/images/focalstack/stack2.png", 0)
	img3 = cv2.imread("cpLabs/static/images/focalstack/stack3.png", 0)
	(height, width) = img1.shape

	images = [img1,img2,img3]
	g_simple = [compute_gradients(img1), compute_gradients(img2), compute_gradients(img3)]
	g_sobel = [compute_sobel_gradients(img1), compute_sobel_gradients(img2), compute_sobel_gradients(img3)]
	
	contrib_simple = [np.zeros((height, width), np.uint8), np.zeros((height, width), np.uint8), np.zeros((height, width), np.uint8)]
	contrib_sobel = [np.zeros((height, width), np.uint8), np.zeros((height, width), np.uint8), np.zeros((height, width), np.uint8)]

	output_simple = np.zeros((height, width), np.uint8)
	for y in range(0, height):
		for x in range(0, width):
			argmax = __argmax([g_simple[0][y,x], g_simple[1][y,x], g_simple[2][y,x]])
			output_simple[y,x] = images[argmax][y,x]
			contrib_simple[argmax][y,x] = 255

	output_sobel = np.zeros((height, width), np.uint8)
	for y in range(0, height):
		for x in range(0, width):
			argmax = __argmax([g_sobel[0][y,x], g_sobel[1][y,x], g_sobel[2][y,x]])
			output_sobel[y,x] = images[argmax][y,x]
			contrib_sobel[argmax][y,x] = 255
	
	# display
	images_disp = np.zeros((height, 3*width), np.uint8)
	images_disp[:height, :width] = img1
	images_disp[:height, width:2*width] = img2
	images_disp[:height, 2*width:3*width] = img3

	simple_disp = np.zeros((height, 4*width), np.uint8)
	simple_disp[:height, :width] = contrib_simple[0]
	simple_disp[:height, width:2*width] = contrib_simple[1]
	simple_disp[:height, 2*width:3*width] = contrib_simple[2]
	simple_disp[:height, 3*width:4*width] = output_simple

	sobel_disp = np.zeros((height, 4*width), np.uint8)
	sobel_disp[:height, :width] = contrib_sobel[0]
	sobel_disp[:height, width:2*width] = contrib_sobel[1]
	sobel_disp[:height, 2*width:3*width] = contrib_sobel[2]
	sobel_disp[:height, 3*width:4*width] = output_sobel

	cv2.imshow("Original images", images_disp)
	cv2.imshow('Output (simple gradient operator)',simple_disp)
	cv2.imshow('Output (sobel operator)', sobel_disp)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def __argmax(array):
	index, element = max(enumerate(array), key=itemgetter(1))
	return index;

main()