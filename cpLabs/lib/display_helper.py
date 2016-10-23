import numpy as np

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
