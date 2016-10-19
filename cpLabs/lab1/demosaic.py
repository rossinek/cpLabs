import cv2
import numpy as np

def demosaic(img, pattern_shift=[1,0]):
	if(len(img.shape) != 3):
		raise NotImplementedError, "demosaic for one-channel images is not implemented"

	(height, width, channels) = img.shape
	
	for y in range(0, height):
		for x in range(0, width):
			if __is_green_px(y,x,pattern_shift):
				img[y,x,0] = img[y,x,2] = 0
			elif __is_red_row(y,x,pattern_shift):
				img[y,x,0] = img[y,x,1] = 0
			else:
				img[y,x,1] = img[y,x,2] = 0

	# filling zeros for each color

	out = img[:,:,1].astype(int) * 4
	out[:-1,:] += img[1:,:,1]
	out[1:,:] += img[:-1,:,1]
	out[:,:-1] += img[:,1:,1]
	out[:,1:] += img[:,:-1,1]
	img[1:-1,1:-1,1] = out[1:-1,1:-1] / 4

	
	out = img[:,:,0].astype(int) * 2
	out[:-1,:] += img[1:,:,0]
	out[1:,:] += img[:-1,:,0]
	out[:,:-1] += img[:,1:,0]
	out[:,1:] += img[:,:-1,0]
	out = out * 2
	out[:-1,:-1] += img[1:,1:,0]
	out[1:,1:] += img[:-1,:-1,0]
	out[1:,:-1] += img[:-1,1:,0]
	out[:-1,1:] += img[1:,:-1,0]
	img[1:-1,1:-1,0] = out[1:-1,1:-1] / 4

	out = img[:,:,2].astype(int) * 2	
	out[:-1,:] += img[1:,:,2]
	out[1:,:] += img[:-1,:,2]
	out[:,:-1] += img[:,1:,2]
	out[:,1:] += img[:,:-1,2]
	out = out * 2
	out[:-1,:-1] += img[1:,1:,2]
	out[1:,1:] += img[:-1,:-1,2]
	out[1:,:-1] += img[:-1,1:,2]
	out[:-1,1:] += img[1:,:-1,2]
	img[1:-1,1:-1,2] = (out[1:-1,1:-1] / 4).astype('uint8')


def slow_demosaic(img, pattern_shift=[1,0]):
	if(len(img.shape) != 3):
		raise NotImplementedError, "demosaic for one-channel images is not implemented"

	(height, width, channels) = img.shape
	
	for y in range(1, height-1):
		for x in range(1, width-1):
			if __is_green_px(y, x, pattern_shift):
				if __is_red_row(y, x, pattern_shift):
					img[y,x,2] = (int(img[y,x-1,2]) + int(img[y,x+1,2])) / 2
					img[y,x,0] = (int(img[y-1,x,0]) + int(img[y+1,x,0])) / 2
				else:
					img[y,x,2] = (int(img[y-1,x,2]) + int(img[y+1,x,2])) / 2
					img[y,x,0] = (int(img[y,x-1,0]) + int(img[y,x+1,0])) / 2
			elif __is_red_row(y, x, pattern_shift):
				img[y,x,1] = (int(img[y-1,x,1]) + int(img[y+1,x,1]) + int(img[y,x-1,1]) + int(img[y,x+1,1])) / 4
				img[y,x,0] = (int(img[y-1,x-1,0]) + int(img[y-1,x+1,0]) + int(img[y+1,x-1,0]) + int(img[y+1,x+1,0])) / 4
			else:
				img[y,x,1] = (int(img[y-1,x,1]) + int(img[y+1,x,1]) + int(img[y,x-1,1]) + int(img[y,x+1,1])) / 4
				img[y,x,2] = (int(img[y-1,x-1,2]) + int(img[y-1,x+1,2]) + int(img[y+1,x-1,2]) + int(img[y+1,x+1,2])) / 4

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