import cv2
import numpy as np

from cpLabs.lib.display_helper import multiple_display
from cpLabs.lib.hdr import bilateral_filter_tone_mapping

def main():
	print "Task 3: Tone mapping"
	image = cv2.imread("cpLabs/static/images/hdr/memorial.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
	
	output1 = bilateral_filter_tone_mapping(image, -1, 0.4, 12)
	output2 = bilateral_filter_tone_mapping(image, -1, 0.5, 12)
	output3 = bilateral_filter_tone_mapping(image, -1, 0.6, 12)

	cv2.imshow('Image1', multiple_display([[image, output1, output2, output3]]))

	cv2.waitKey(0)
	cv2.destroyAllWindows()

main()