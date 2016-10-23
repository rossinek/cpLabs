import cv2
import numpy as np

from cpLabs.lib.display_helper import multiple_display
from cpLabs.lib.focus_stack import *

def main():
	print "Task 1: Focal Stack"
	images = [
		cv2.imread("cpLabs/static/images/focalstack/stack1.png", 0),
		cv2.imread("cpLabs/static/images/focalstack/stack2.png", 0),
		cv2.imread("cpLabs/static/images/focalstack/stack3.png", 0)
	]

	output_simple, contrib_simple = all_in_focus(images, SIMPLE_OPERATOR_FLAG)
	output_sobel, contrib_sobel = all_in_focus(images, SOBEL_OPERATOR_FLAG)

	display_array = [
		images,
		[contrib_simple[0], contrib_simple[1], contrib_simple[2], output_simple],
		[contrib_sobel[0], contrib_sobel[1], contrib_sobel[2], output_sobel]
	]
	
	all_in_one_display = multiple_display(display_array)
	cv2.imshow('original, simple, sobel', all_in_one_display)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



main()