import cv2
import numpy as np

from cpLabs.lib.display_helper import multiple_display

def main():
	print "Task 3: Tone mapping"
	image = cv2.imread("cpLabs/static/images/hdr/memorial.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
	(height, width, channels) = image.shape

	output = image.copy()
	input_intensity = (20.0*image[:,:,2]+40.0*image[:,:,1]+image[:,:,0])/61.0
	
	log_input_intensity = np.log10(input_intensity)	
	log_base = cv2.bilateralFilter(log_input_intensity, -1, 0.8, 12)
	compressionFactor = np.log10(5)/(np.amax(log_base) - np.amin(log_base))
	log_absolute_scale = np.amax(log_base)*compressionFactor
	log_detail = log_input_intensity - log_base
	log_output_intensity = log_base*compressionFactor + log_detail - log_absolute_scale

	for c in range(0, channels):
		value = image[:,:,c]/input_intensity
		output[:,:,c] = value*10.0**log_output_intensity

	print np.amin(image), np.amax(image)
	print np.amin(output), np.amax(output)

	cv2.imshow('Image1', multiple_display([[image, output]]))
	#cv2.imshow('Image', multiple_display([[log_input_intensity, log_base]]))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def __clip(x, x_min, x_max):
	return min(max(x, x_min), x_max)

main()