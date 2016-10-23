import cv2
import numpy as np

from cpLabs.lib.display_helper import multiple_display

def main():
	print "Task 3: Tone mapping"
	image = cv2.imread("cpLabs/static/images/hdr/memorial.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
	(height, width, channels) = image.shape
	
	output = image.copy()
	input_intensity = np.zeros((height,width), np.float32)
	for y in range(0, height):
		for x in range(0, width):
			input_intensity[y,x] = (20.0*image[y,x,2]+40.0*image[y,x,1]+image[y,x,0])/61.0
	
	log_input_intensity = np.log10(input_intensity)
	bilateral_log_input_intensity = cv2.bilateralFilter(log_input_intensity, -1, 0.2, 6)

	compressionFactor = np.log10(6.0)/(np.amax(bilateral_log_input_intensity) - np.amin(bilateral_log_input_intensity))
	log_absolute_scale = np.amax(bilateral_log_input_intensity)*compressionFactor

	for y in range(0, height):
		for x in range(0, width):
			for c in range(0, channels):
				value=image[y,x,c]/(input_intensity[y,x]);
				log_base = bilateral_log_input_intensity[y,x]
				log_detail=log_input_intensity[y,x] - log_base
				log_output_intensity=log_base*compressionFactor + log_detail - log_absolute_scale
				output[y,x,c] = __clip(value*10.0**log_output_intensity, 0.0, 255.0)
	
	print output
	cv2.imshow('Image1', multiple_display([[image, output]]))

	cv2.imshow('Image', multiple_display([[log_input_intensity, bilateral_log_input_intensity]]))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def __clip(x, x_min, x_max):
	return min(max(x, x_min), x_max)

main()