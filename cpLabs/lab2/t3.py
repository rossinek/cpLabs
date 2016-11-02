import cv2
import numpy as np

from cpLabs.lib.display_helper import multiple_display
from cpLabs.lib.hdr import bilateral_filter_tone_mapping

def main():
	print "Task 3: Tone mapping"
	image = cv2.imread("cpLabs/static/images/hdr/memorial.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
	image2 = cv2.imread("output/myhdr.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

	output_1_1 = bilateral_filter_tone_mapping(image, -1, 0.2, 12, 8)
	output_1_2 = bilateral_filter_tone_mapping(image, -1, 0.5, 12, 8)
	output_1_3 = bilateral_filter_tone_mapping(image, -1, 0.8, 12, 8)

	output_2_1 = bilateral_filter_tone_mapping(image2, -1, 0.2, 4, 200)
	output_2_2 = bilateral_filter_tone_mapping(image2, -1, 0.7, 4, 200)
	output_2_3 = bilateral_filter_tone_mapping(image2, -1, 3.2, 4, 200)

	cv2.imshow('Image1', multiple_display([[output_1_1, output_1_2, output_1_3]]))
	cv2.imshow('Image2', multiple_display([[output_2_1, output_2_2, output_2_3]]))

	cv2.waitKey(0)
	cv2.destroyAllWindows()

main()