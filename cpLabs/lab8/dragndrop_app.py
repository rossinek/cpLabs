import cv2
import numpy as np

from lib.poisson import poisson_conj, naive_composite
import sys


if len(sys.argv) != 3:
	sys.exit("Usage: %s <bg_path> <fg_path>" % sys.argv[0])

bg_path = sys.argv[1]
fg_path = sys.argv[2]

mousedown = False
mode = False 		# drawing / moving
bx, by = -1, -1 	# mouse position on keydown
lx, ly = -1, -1 	# previous mouse position
fgx, fgy = 0, 0 	# position of fg
scale = 1.0 		# fg scale

bg = cv2.imread(bg_path)
_fg = cv2.imread(fg_path)
resize_factor = min(float(bg.shape[0])/_fg.shape[0], float(bg.shape[1])/_fg.shape[1])
if resize_factor < 1:
	_fg = cv2.resize(_fg, (int(_fg.shape[1]*resize_factor),int(_fg.shape[0]*resize_factor)))
fg = _fg.copy()

mask = np.zeros(fg.shape, np.uint8)


# mouse callback function
def mouse_paint(event,x,y,flags,param):
	global lx, ly, bx, by, mousedown, mode, fgx, fgy, mask, _fg, fg, bg

	if event == cv2.EVENT_LBUTTONDOWN:
		mousedown = True
		bx, by = x, y
		if mode == False:
			mask = np.zeros(fg.shape, np.uint8)

	elif event == cv2.EVENT_MOUSEMOVE:
		if mousedown == True:
			if mode == False:
				posx = min(max(0, x-fgx), fg.shape[1])
				posy = min(max(0, y-fgy), fg.shape[0])
				lposx = min(max(0, lx-fgx), fg.shape[1])
				lposy = min(max(0, ly-fgy), fg.shape[0])
				cv2.circle(mask,(posx,posy),5,(255,255,255),-1)
				cv2.line(mask, (lposx,lposy), (posx,posy), (255,255,255), 5)
			else:
				fgx = min(max(-fg.shape[1], fgx+(x-lx)), bg.shape[1]+_fg.shape[1])
				fgy = min(max(-fg.shape[0], fgy+(y-ly)), bg.shape[0])

	elif event == cv2.EVENT_LBUTTONUP:
		mousedown = False
		if mode == False:
			posx = min(max(0, x-fgx), fg.shape[1])
			posy = min(max(0, y-fgy), fg.shape[0])

			bposx = min(max(0, bx-fgx), fg.shape[1])
			bposy = min(max(0, by-fgy), fg.shape[0])
			cv2.circle(mask,(posx,posy),5,(255,255,255),-1)
			cv2.line(mask, (bposx,bposy), (posx,posy), (255,255,255), 5)
			floodfill_mask()
		else:
			fgx = min(max(-fg.shape[1], fgx+(x-lx)), bg.shape[1]+_fg.shape[1])
			fgy = min(max(-fg.shape[0], fgy+(y-ly)), bg.shape[0])

	lx, ly = x, y

def floodfill_mask():
	global mask
	h, w = mask.shape[:2]
	mcp = np.zeros((h+2, w+2), np.uint8)
	mcp[1:-1,1:-1] = mask[...,0].copy()
	m = np.zeros((h+4, w+4), np.uint8)
	cv2.floodFill(mcp, m, (0,0), 255)
	mask[...] = (1-np.repeat(m[...,None], 3, axis=-1)[2:-2,2:-2])*255

def state_display():
	global _fg, fg, bg, mask, fgx, fgy

	(bh,bw,bc) = bg.shape
	(fh,fw,fc) = _fg.shape
	out = np.zeros((bh, bw+fw, bc), np.uint8)
	out[:, fw:] = bg[...]
	
	_f, m = stretch_n_crop(out.shape, fg, (fgx,fgy))
	_m, _ = stretch_n_crop(out.shape, mask, (fgx,fgy))
	bmask = _m.astype(np.bool)

	out[m] = _f[m]*0.5 + out[m]*0.5
	out[bmask] = _f[bmask]

	return out

def stretch_n_crop(shape, img, pos):
	(h,w) = shape[:2]
	(x,y) = pos
	out = np.zeros(shape, img.dtype)
	m = np.zeros(shape, img.dtype)
	ax0, ay0 = max(0,-x), max(0,-y)
	bx0, by0 = max(0,x), max(0,y)
	aw = img.shape[1] - ax0
	ah = img.shape[0] - ay0
	bw = w - bx0
	bh = h - by0
	ax1 = ax0 + aw
	ay1 = ay0 + ah
	bx1 = bx0 + bw
	by1 = by0 + bh
	if bh>0 and bw>0 and ah>0 and aw>0:
		if aw > bw:
			ax1 = ax0 + bw
		else:
			bx1 = bx0 + aw
		if ah > bh:
			ay1 = ay0 + bh
		else:
			by1 = by0 + ah
		out[by0:by1,bx0:bx1] = img[ay0:ay1,ax0:ax1]
		m[by0:by1,bx0:bx1] = 255
	return out, m.astype(np.bool)

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_paint)

while(1):
	cv2.imshow('image',state_display())
	k = cv2.waitKey(1) & 0xFF
	if k == ord('m'):
		mode = not mode
	elif k == ord('='): # key =/+
		if scale < 2.2:
			scale += 0.1
			fg = cv2.resize(_fg, (int(_fg.shape[1]*scale),int(_fg.shape[0]*scale)))
			mask = cv2.resize(mask, (int(_fg.shape[1]*scale),int(_fg.shape[0]*scale)))			
	elif k == ord('-'): # key -/_
		if scale > 0.2:
			scale -= 0.1
			fg = cv2.resize(_fg, (int(_fg.shape[1]*scale),int(_fg.shape[0]*scale)))
			mask = cv2.resize(mask, (int(_fg.shape[1]*scale),int(_fg.shape[0]*scale)))			
	elif (k == 13) or (k == 32):
		(fh,fw,fc) = _fg.shape
		(bh,bw,bc) = bg.shape
		fg_v, _ = stretch_n_crop(bg.shape, fg, (fgx-fw,fgy))
		mask_v, _ = stretch_n_crop(bg.shape, mask, (fgx-fw,fgy))
		
		disp = np.zeros((bh, bw+fw, bc), np.uint8)
		disp[:, fw:] = poisson_conj(bg,fg_v,mask_v, 200)

		cv2.imshow('image',disp)
		cv2.waitKey(0)
	elif k == 27:
		break

cv2.destroyAllWindows()