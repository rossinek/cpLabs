import numpy as np
import cv2
from lib.display_helper import *

def display_cups_board_color_output(image, outputs, kernels, path):
    crop_area1 = np.array([[250, 400], [650, 800]])
    crop_area2 = np.array([[850, 1000], [850, 1000]])
    outputs_crop1 = outputs[:,crop_area1[0,0]:crop_area1[0,1],crop_area1[1,0]:crop_area1[1,1]].copy()
    outputs_crop2 = outputs[:,crop_area2[0,0]:crop_area2[0,1],crop_area2[1,0]:crop_area2[1,1]].copy()

    outdisp1 = []
    outdisp2 = []
    for i in range(len(kernels)):
        k = kernels[i]
        kernel_disp = (k+np.min(k)) * ( 255./(np.max(k)+np.min(k)))
        kernel_disp = np.repeat(kernel_disp[...,None], 3, axis=-1)
        out_img1 = outputs_crop1[i]
        out_img2 = outputs_crop2[i]
        out_img1[:k.shape[0], :k.shape[1]] = kernel_disp
        out_img2[:k.shape[0], :k.shape[1]] = kernel_disp
        outdisp1.append(out_img1)
        outdisp2.append(out_img2)

    img_crop1 = image[crop_area1[0,0]:crop_area1[0,1],crop_area1[1,0]:crop_area1[1,1]].copy()
    img_crop2 = image[crop_area2[0,0]:crop_area2[0,1],crop_area2[1,0]:crop_area2[1,1]].copy()
    outdisp1.append(img_crop1)
    outdisp2.append(img_crop2)

    outdisp = outdisp1 + outdisp2

    outdisp = [outdisp[i*5:i*5+5] for i in range(int(np.ceil(len(outdisp)/5.)))]

    output_crop_display = multiple_display(outdisp)
    cv2.imwrite(path, output_crop_display)
