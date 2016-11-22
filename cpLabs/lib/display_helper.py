import numpy as np
import cv2

def multiple_display(array):
	sizes = []
	disp_shape = list(array[0][0].shape)
	disp_shape[0] = disp_shape[1] = 0
	for y in range(0, len(array)):
		width = 0
		height = 0
		for x in range(0, len(array[y])):
			width += array[y][x].shape[1]
			height = max(height, array[y][x].shape[0])
		sizes.append([height, width])
		disp_shape[1] = max(disp_shape[1], width)
		disp_shape[0] = disp_shape[0] + height

	output_display = np.zeros(tuple(disp_shape), array[0][0].dtype)

	row_pos = 0
	for y in range(0, len(array)):
		col_pos = 0
		for x in range(0, len(array[y])):
			h, w = array[y][x].shape[0], array[y][x].shape[1]
			output_display[row_pos:row_pos+h,col_pos:col_pos+w] = array[y][x]
			col_pos += w
		row_pos+=sizes[y][0]

	return output_display

def draw_point(img, A):
	cv2.rectangle(img, (A[0]-1,A[1]-1), (A[0]+1,A[1]+1), (0,255,0), -1)

def draw_arrow(img, A, B):
	cv2.line(img, A, B, (255,0,0), 1, 8)
	cv2.rectangle(img, (A[0]-1,A[1]-1), (A[0]+1,A[1]+1), (0,255,0), -1)
	cv2.rectangle(img, (B[0]-1,B[1]-1), (B[0]+1,B[1]+1), (0,0,255), -1)

def draw_circle_on_center(img):
	(h, w, c) = img.shape
	center = (w/2, h/2)
	radius = min(w/2, h/2)
	cv2.circle(img, center, radius, (255, 255, 255))

def draw_rectangle_on_center(img):
	(h, w, c) = img.shape
	center = (w/2, h/2)
	rw = w
	rh = w/2
	cv2.rectangle(img, (0,h/2-rh/2), (w-1,h/2+rh/2), (255, 255, 255))

def normalized_copy(img):
	return img*(255./np.max(img))
