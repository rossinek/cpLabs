import numpy as np

def gamma_correction(img, gamma):
	gc = 1.0 / gamma
	img[...] = 255.0*(img/255.0)**gc

def linearise_srgb(img):
	vecfunc_linearize = np.vectorize(__linearize_srgb_value)
	return vecfunc_linearize(img)

def __linearize_srgb_value(v):
    if (v<=0.04045): return v/12.92
    else: return ((v+0.055)/1.055)**2.4