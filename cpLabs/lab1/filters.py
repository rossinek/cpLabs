import cv2
import numpy as np

def median_filter(img, channel, f_height, f_width):
	if f_width%2==0 or f_height%2==0:
		raise NotImplementedError, "demosaic implemented only for three-channel images"
	if f_width == 1 and f_height == 1:
		return

	halfH = f_height/2
	halfW = f_width/2
	center = (f_height*f_width)/2
	img_wide = cv2.copyMakeBorder(img,halfH,halfH,halfW,halfW,cv2.BORDER_REFLECT_101)
	(height, width, channels) = img_wide.shape

	for y in range(halfH, height-halfH):
		for x in range(halfW, width-halfW):
			neighborhood = np.hstack(img_wide[y-halfH:y+halfH+1, x-halfW:x+halfW+1, channel])
			neighborhood.sort()
			img_wide[y,x,channel] = neighborhood[center]
	
	img[:,:] = img_wide[halfH:-halfH,halfW:-halfW]

def median_filter_chromacity(img, f_height, f_width):
	img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	median_filter(img_YCrCb, 1, f_height, f_width)
	median_filter(img_YCrCb, 2, f_height, f_width)
	img_BGR = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCR_CB2BGR)
	img[:,:] = img_BGR[:,:]