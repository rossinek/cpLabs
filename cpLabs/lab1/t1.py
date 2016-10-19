import cv2
from cpLabs.lab1.demosaic import basic_demosaic, slow_basic_demosaic, slow_edge_based_demosaic
from cpLabs.lab1.gamma_correction import gamma_correction

def main():
	img1 = cv2.imread('cpLabs/static/images/lighthouse_RAW_noisy_sigma0.01.png')
	slow_basic_demosaic(img1)
	gamma_correction(img1, 2.2)
	cv2.imshow('image',img1)

	img2 = cv2.imread('cpLabs/static/images/lighthouse_RAW_noisy_sigma0.01.png')
	slow_edge_based_demosaic(img2)
	gamma_correction(img2, 2.2)
	cv2.imshow('image2',img2)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()

main()