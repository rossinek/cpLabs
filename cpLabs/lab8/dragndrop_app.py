import cv2
import numpy as np

mousedown = False
mode = False 	# drawing / moving
bx, by = -1, -1 # mouse position on keydown
lx, ly = -1, -1 # previous mouse position
fgx, fgy = 0, 0 # position of fg

fg = cv2.imread('data/bear.png')
bg = cv2.imread('data/waterpool.png')

resize_factor = min(float(bg.shape[0])/fg.shape[0], float(bg.shape[1])/fg.shape[1])

if resize_factor < 1:
	fg = cv2.resize(fg, (fg.shape[1]*resize_factor,fg.shape[0]*resize_factor))

mask = np.zeros(fg.shape, np.uint8)


# mouse callback function
def mouse_paint(event,x,y,flags,param):
	global lx, ly, bx, by, mousedown, mode, fgx, fgy, mask, fg, bg

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
				fgx = min(max(0, fgx+(x-lx)), bg.shape[1])
				fgy = min(max(0, fgy+(y-ly)), bg.shape[0]-fg.shape[0])

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
			fgx = min(max(0, fgx+(x-lx)), bg.shape[1])
			fgy = min(max(0, fgy+(y-ly)), bg.shape[0]-fg.shape[0])

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
	global fg, bg, mask, fgx, fgy

	bmask = mask.astype(np.bool)
	(bh,bw,bc) = bg.shape
	(fh,fw,fc) = fg.shape

	out = np.zeros((bh, bw+fw, bc), np.uint8)
	out[:, fw:] = bg[...]
	
	out[fgy:fgy+fh, fgx:fgx+fw] = (fg[...]*0.5 + out[fgy:fgy+fh, fgx:fgx+fw]*0.5).clip(0, 255)
	out[fgy:fgy+fh, fgx:fgx+fw] = fg[...]*bmask+out[fgy:fgy+fh, fgx:fgx+fw]*(1-bmask)

	return out



cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_paint)

while(1):
	cv2.imshow('image',state_display())
	k = cv2.waitKey(1) & 0xFF
	if k == ord('m'):
		mode = not mode
	elif (k == 13) or (k == 32):
		
		cv2.imshow('image',state_display())
		cv2.waitKey(0)
		break
	elif k == 27:
		break

cv2.destroyAllWindows()