import cv2
import numpy as np

#from cpLabs.lib.display_helper import multiple_display
from cpLabs.lib.hdr import simple_robertson

def main():
	print "Task 2: HDR"
	images = [
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0061.png"),
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0062.png"),
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0063.png"),
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0064.png"),
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0065.png"),
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0066.png"),
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0067.png"),
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0068.png"),
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0069.png"),
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0070.png"),
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0071.png"),
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0072.png"),
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0073.png"),
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0074.png"),
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0075.png"),
		cv2.imread("cpLabs/static/images/Memorial_SourceImages/memorial0076.png")
	]

	exposures = [
		1.0/0.03125,
		1.0/0.0625,
		1.0/0.125,
		1.0/0.25,
		1.0/0.5,
		1.0/1.0,
		1.0/2.0,
		1.0/4.0,
		1.0/8.0,
		1.0/16.0,
		1.0/32.0,
		1.0/64.0,
		1.0/128.0,
		1.0/256.0,
		1.0/512.0,
		1.0/1024.0
	]
	
	output = simple_robertson(images, exposures)

	cv2.imwrite('output/memorial.exr',output/255.)
	
	#cv2.imshow('Image1', output/255.)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

main()