import cv2
import numpy as np

#from cpLabs.lib.display_helper import multiple_display
from cpLabs.lib.hdr import simple_robertson

def main():
	print "Task 2: HDR"

	images = [
		cv2.imread("cpLabs/static/images/my_hdr/_DSC00051.JPG"),
		cv2.imread("cpLabs/static/images/my_hdr/_DSC00052.JPG"),
		cv2.imread("cpLabs/static/images/my_hdr/_DSC00053.JPG"),
		cv2.imread("cpLabs/static/images/my_hdr/_DSC00054.JPG"),
		cv2.imread("cpLabs/static/images/my_hdr/_DSC00055.JPG")
	]

	exposures = [
		1.0/50.0,
		1.0/80.0,
		1.0/30.0,
		1.0/125.0,
		1.0/20.0
	]
	
	output = simple_robertson(images, exposures)

	cv2.imwrite('output/myhdr.exr',output/255.)

main()