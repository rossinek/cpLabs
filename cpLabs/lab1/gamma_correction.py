
def gamma_correction(img, gamma):
	if(len(img.shape) != 3):
		(height, width) = img.shape
		channels = 1
	else:
		(height, width, channels) = img.shape

	gc = 1.0 / gamma
	for y in range(0, height):
		for x in range(0, width):
			for c in range(0,channels):
				img[y,x,c] = int(255.0*(float(img[y,x,c])/255.0)**gc)
