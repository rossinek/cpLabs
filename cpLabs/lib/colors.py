import cv2
import numpy as np

def transfer_color(img_src, img_dest):
    if len(img_dest.shape) == 2:
        (h,w) = img_dest.shape
        img_dest_valid = np.repeat(img_dest[...,None], 3, axis=-1)
    elif len(img_dest.shape) == 3:
        (h,w,c) = img_dest.shape
        img_dest_valid = img_dest.copy()
    else:
        raise NotImplementedError("Implemented for img_dest with shape (h,w,c) or (h,w)")
    if(img_src.shape[:2] != img_dest.shape[:2]):
        img_src_valid = cv2.resize(img_src, (w, h))
    else:
        img_src_valid = img_src
    img_src_YCrCb = cv2.cvtColor(img_src_valid, cv2.COLOR_BGR2YCR_CB)
    out_YCrCb = cv2.cvtColor(img_dest_valid, cv2.COLOR_BGR2YCR_CB)
    out_YCrCb[:,:,1] = img_src_YCrCb[:,:,1]
    out_YCrCb[:,:,2] = img_src_YCrCb[:,:,2]
    out_BGR = cv2.cvtColor(out_YCrCb, cv2.COLOR_YCR_CB2BGR)
    return out_BGR