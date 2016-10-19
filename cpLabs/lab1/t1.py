# from cpLabs.lab1.demosaic import
import cv2
from cpLabs.lab1.demosaic import basic_demosaic, slow_basic_demosaic

def main():
	img = cv2.imread('cpLabs/static/images/lighthouse_RAW_noisy_sigma0.01.png')
	#img = cv2.imread('cpLabs/static/images/raw/text.png')
	basic_demosaic(img, [0,1])
	cv2.imshow('image',img)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()

main()