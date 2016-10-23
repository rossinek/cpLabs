import cv2
import numpy as np

# Arguments diameter, sigma_color and sigma_space are for OpenCV bilateral filter
#
# diameter 
#	-	Diameter of each pixel neighborhood that is used during filtering. 
#		If it is non-positive, it is computed from sigma_space.
# sigmaColor
#	-	Filter sigma in the color space. 
#		A larger value of the parameter means that farther colors 
#		within the pixel neighborhood (see sigma_space) will be mixed together, 
#		resulting in larger areas of semi-equal color.
# sigmaSpace 
#	-	Filter sigma in the coordinate space. 
#		A larger value of the parameter means that farther pixels will influence
#		each other as long as their colors are close enough (see sigmaColor). 
#		When d>0, it specifies the neighborhood size regardless of sigmaSpace.
#		Otherwise, d is proportional to sigmaSpace.
# from docs: http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?#bilateralfilter

def bilateral_filter_tone_mapping(hdr_img, diameter, sigma_color, sigma_space):
	(height, width, channels) = hdr_img.shape
	output = hdr_img.copy()

	input_intensity = (20.0*hdr_img[:,:,2]+40.0*hdr_img[:,:,1]+hdr_img[:,:,0])/61.0
	log_input_intensity = np.log10(input_intensity)	
	log_base = cv2.bilateralFilter(log_input_intensity, diameter, sigma_color, sigma_space)
	compressionFactor = np.log10(5)/(np.amax(log_base) - np.amin(log_base))
	log_absolute_scale = np.amax(log_base)*compressionFactor
	log_detail = log_input_intensity - log_base
	log_output_intensity = log_base*compressionFactor + log_detail - log_absolute_scale

	for c in range(0, channels):
		value = hdr_img[:,:,c]/input_intensity
		output[:,:,c] = value*10.0**log_output_intensity

	return output
