import numpy as np
import cv2

from scipy.signal import convolve2d

def naive_composite(bg, fg, mask):
    bmask = mask.astype(np.bool)
    output = bg.copy()
    output[bmask] = fg[bmask]
    return output

def laplacian(img):
    l_operator = np.array([[0., -1., 0. ],[-1. , 4., -1.],[0., -1., 0.]], np.float64)
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[...,c] = convolve2d(img[...,c], l_operator, mode='same')
    return out
    
def poisson_gdesc(bg, fg, mask, max_it):
    if not (bg.shape == fg.shape and fg.shape == mask.shape):
        raise NotImplementedError("Implemented only for equal shapes")
    bmask = mask.astype(np.bool)
    b = laplacian(fg.astype(np.float64))*bmask
    x = bg.astype(np.float64)
    x[bmask] = 0.0
    for iteration in range(max_it):
        Ax = laplacian(x)
        r = (b - Ax) * bmask
        r_flat = r.flatten()
        Ar_flat = laplacian(r).flatten()
        alpha = np.dot(r_flat, r_flat) / np.dot(r_flat, Ar_flat)
        x = x + (r*bmask) * alpha
    return x.clip(0,255).astype(bg.dtype)

def poisson_conj(bg, fg, mask, max_it):
    if not (bg.shape == fg.shape and fg.shape == mask.shape):
        raise NotImplementedError("Implemented only for equal shapes")
    bmask = mask.astype(np.bool)
    b = laplacian(fg.astype(np.float64)) * bmask
    x = bg.astype(np.float64)
    x[bmask] = 0.0
    Ax = laplacian(x) * bmask
    r = b - Ax
    d = r.copy()
    for iteration in range(max_it):
        Ad = laplacian(d) * bmask
        r_dot = np.dot(r.flatten(), r.flatten())
        alpha = r_dot / np.dot(d.flatten(), Ad.flatten())
        x += d*bmask*alpha
        r_new = (r-alpha*Ad) * bmask
        beta = np.dot(r_new.flatten(), r_new.flatten()) / r_dot
        d = (r_new + beta*d) * bmask
        r = r_new
    return x.clip(0,255).astype(bg.dtype)