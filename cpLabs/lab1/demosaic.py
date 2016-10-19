import cv2
import numpy as np

def linear_demosaic(img, pattern_shift=[1,0]):
	if(len(img.shape) != 3 or img.shape[2] != 3):
		raise NotImplementedError, "demosaic implemented only for three-channel images"

	(height, width, channels) = img.shape
	
	# filling zeros for each color
	for y in range(0, height):
		for x in range(0, width):
			color = __get_px_color(y,x,pattern_shift)
			img[y,x,(color+1)%3] = img[y,x,(color-1)%3] = 0

	img_wide = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT_101)
	
	out = img_wide[:,:,1].astype(int) * 4
	out[:-1,:] += img_wide[1:,:,1]
	out[1:,:] += img_wide[:-1,:,1]
	out[:,:-1] += img_wide[:,1:,1]
	out[:,1:] += img_wide[:,:-1,1]
	img_wide[1:-1,1:-1,1] = out[1:-1,1:-1] / 4
	
	out = img_wide[:,:,0].astype(int) * 2
	out[:-1,:] += img_wide[1:,:,0]
	out[1:,:] += img_wide[:-1,:,0]
	out[:,:-1] += img_wide[:,1:,0]
	out[:,1:] += img_wide[:,:-1,0]
	out = out * 2
	out[:-1,:-1] += img_wide[1:,1:,0]
	out[1:,1:] += img_wide[:-1,:-1,0]
	out[1:,:-1] += img_wide[:-1,1:,0]
	out[:-1,1:] += img_wide[1:,:-1,0]
	img_wide[1:-1,1:-1,0] = out[1:-1,1:-1] / 4

	out = img_wide[:,:,2].astype(int) * 2	
	out[:-1,:] += img_wide[1:,:,2]
	out[1:,:] += img_wide[:-1,:,2]
	out[:,:-1] += img_wide[:,1:,2]
	out[:,1:] += img_wide[:,:-1,2]
	out = out * 2
	out[:-1,:-1] += img_wide[1:,1:,2]
	out[1:,1:] += img_wide[:-1,:-1,2]
	out[1:,:-1] += img_wide[:-1,1:,2]
	out[:-1,1:] += img_wide[1:,:-1,2]
	img_wide[1:-1,1:-1,2] = out[1:-1,1:-1] / 4

	img[:,:] = img_wide[1:-1,1:-1]

def slow_linear_demosaic(img, pattern_shift=[1,0]):
	if(len(img.shape) != 3 or img.shape[2] != 3):
		raise NotImplementedError, "demosaic implemented only for three-channel images"

	(height, width, channels) = img.shape
	
	img_wide = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT_101)
	# shift pattern shift
	wide_pattern_shift = [s+1%2 for s in pattern_shift]

	for y in range(1, height+1):
		for x in range(1, width+1):
			if __is_green_px(y, x, wide_pattern_shift):
				if __is_red_row(y, x, wide_pattern_shift):
					img_wide[y,x,2] = (int(img_wide[y,x-1,2]) + int(img_wide[y,x+1,2])) / 2
					img_wide[y,x,0] = (int(img_wide[y-1,x,0]) + int(img_wide[y+1,x,0])) / 2
				else:
					img_wide[y,x,2] = (int(img_wide[y-1,x,2]) + int(img_wide[y+1,x,2])) / 2
					img_wide[y,x,0] = (int(img_wide[y,x-1,0]) + int(img_wide[y,x+1,0])) / 2
			else:
				img_wide[y,x,1] = (int(img_wide[y-1,x,1]) + int(img_wide[y+1,x,1]) + int(img_wide[y,x-1,1]) + int(img_wide[y,x+1,1])) / 4
				if __is_red_row(y, x, wide_pattern_shift):
					img_wide[y,x,0] = (int(img_wide[y-1,x-1,0]) + int(img_wide[y-1,x+1,0]) + int(img_wide[y+1,x-1,0]) + int(img_wide[y+1,x+1,0])) / 4
				else:
					img_wide[y,x,2] = (int(img_wide[y-1,x-1,2]) + int(img_wide[y-1,x+1,2]) + int(img_wide[y+1,x-1,2]) + int(img_wide[y+1,x+1,2])) / 4
	img[:,:] = img_wide[1:-1,1:-1]

def slow_edge_based_demosaic(img, pattern_shift=[1,0]):
	if(len(img.shape) != 3 or img.shape[2] != 3):
		raise NotImplementedError, "demosaic implemented only for three-channel images"

	(height, width, channels) = img.shape
	
	img_wide = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT_101)
	# shift pattern shift
	wide_pattern_shift = [s+1%2 for s in pattern_shift]

	# interpolate green
	for y in range(1, height+1):
		for x in range(1, width+1):
			if not __is_green_px(y, x, wide_pattern_shift):
				gV = abs(int(img_wide[y-1,x,1])-int(img_wide[y+1,x,1]))
				gH = abs(int(img_wide[y,x-1,1])-int(img_wide[y,x+1,1]))
				if gV == gH:
					img_wide[y,x,1] = (int(img_wide[y-1,x,1]) + int(img_wide[y+1,x,1]) + int(img_wide[y,x-1,1]) + int(img_wide[y,x+1,1])) / 4
				elif gV < gH:
					img_wide[y,x,1] = (int(img_wide[y-1,x,1]) + int(img_wide[y+1,x,1])) / 2
				else:
					img_wide[y,x,1] = (int(img_wide[y,x-1,1]) + int(img_wide[y,x+1,1])) / 2

	# interpolate R and B
	# 	- interpolate R-G or B-G
	#	- add G
	for y in range(1, height+1):
		for x in range(1, width+1):
			if __is_green_px(y, x, wide_pattern_shift):
				if __is_red_row(y, x, wide_pattern_shift):
					red = (int(img_wide[y,x-1,2])-int(img_wide[y,x-1,1])+int(img_wide[y,x+1,2])-int(img_wide[y,x+1,1]))/2 + int(img_wide[y,x,1])
					img_wide[y,x,2] = __constrain_uint8(red)
					blue = (int(img_wide[y-1,x,0])-int(img_wide[y-1,x,1])+int(img_wide[y+1,x,0])-int(img_wide[y+1,x,1]))/2 + int(img_wide[y,x,1])
					img_wide[y,x,0] = __constrain_uint8(blue)
				else:

					blue = (int(img_wide[y,x-1,0])-int(img_wide[y,x-1,1])+int(img_wide[y,x+1,0])-int(img_wide[y,x+1,1]))/2 + int(img_wide[y,x,1])
					img_wide[y,x,0] = __constrain_uint8(blue)
					red = (int(img_wide[y-1,x,2])-int(img_wide[y-1,x,1])+int(img_wide[y+1,x,2])-int(img_wide[y+1,x,1]))/2 + int(img_wide[y,x,1])
					img_wide[y,x,2] = __constrain_uint8(red)

			else:
				if __is_red_row(y, x, wide_pattern_shift):
					blue = (int(img_wide[y-1,x-1,0])-int(img_wide[y-1,x-1,1]) + int(img_wide[y-1,x+1,0])-int(img_wide[y-1,x+1,1]) + int(img_wide[y+1,x-1,0])-int(img_wide[y+1,x-1,1]) + int(img_wide[y+1,x+1,0])-int(img_wide[y+1,x+1,1])) / 4 + int(img_wide[y,x,1])
					img_wide[y,x,0] = __constrain_uint8(blue)
				else:
					red = (int(img_wide[y-1,x-1,2])-int(img_wide[y-1,x-1,1]) + int(img_wide[y-1,x+1,2])-int(img_wide[y-1,x+1,1]) + int(img_wide[y+1,x-1,2])-int(img_wide[y+1,x-1,1]) + int(img_wide[y+1,x+1,2])-int(img_wide[y+1,x+1,1])) / 4 + int(img_wide[y,x,1])
					img_wide[y,x,2] = __constrain_uint8(red)

	img[:,:] = img_wide[1:-1,1:-1]


def __is_green_px(y, x, pattern_shift):
	return ((x+y)%2) == ((pattern_shift[0]+pattern_shift[1])%2)

def __is_red_px(y, x, pattern_shift):
	return not __is_green_px(y, x, pattern_shift) and (y%2) == (pattern_shift[1]%2)

def __is_blue_px(y, x, pattern_shift):
	return not __is_green_px(y, x, pattern_shift) and (y%2) != (pattern_shift[1]%2)

def __is_red_row(y, x, pattern_shift):
	return (y%2) == (pattern_shift[1]%2)

def __is_blue_row(y, x, pattern_shift):
	return (y%2) != (pattern_shift[1]%2)

def __get_px_color(y, x, pattern_shift):
	if __is_green_px(y,x,pattern_shift):
		return 1
	elif __is_red_row(y,x,pattern_shift):
		return 2
	else:
		return 0

def __constrain(v, min_v, max_v):
	return min(max(v,min_v), max_v)

def __constrain_uint8(v):
	return min(max(v,0), 255)