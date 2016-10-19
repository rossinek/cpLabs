import cv2
from cpLabs.lab1.demosaic import basic_demosaic, slow_basic_demosaic, slow_edge_based_demosaic

def main():
	img1 = cv2.imread('cpLabs/static/images/raw/signs-small.png')
	slow_basic_demosaic(img1, [0,1])
	cv2.imshow('image',img1)

	img2 = cv2.imread('cpLabs/static/images/raw/signs-small.png')
	slow_edge_based_demosaic(img2, [0,1])
	cv2.imshow('image2',img2)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()

main()