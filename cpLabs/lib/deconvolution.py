import numpy as np
import cv2
import scipy.io

def compute_prior_L2_frequency(I):
    (n, m, c) = I.shape
    Gx = np.fft.fft2(np.array([[-1,1]]), (n,m))
    Gy = np.fft.fft2(np.array([[-1],[1]]), (n,m))
    Gx = np.repeat(Gx[...,None], c, axis=-1)
    Gy = np.repeat(Gy[...,None], c, axis=-1)
    return Gx, Gy

# kernel size is expected to be odd in both dimensions 
def deconvolution_L2_frequency(I, F, Gx, Gy, weight):
    A = np.conj(F)*F + weight*(np.conj(Gx)*Gx + np.conj(Gy)*Gy);
    b = np.conj(F)*I;
    X = b/A;
    
    x = np.fft.ifft2(X, axes=(-3, -2));
    return x.real.clip(0,255).astype(np.uint8)

def flip_kernel(kernel):
    return np.fliplr(np.flipud(kernel))

def deconvolution_L2(image, kernels):
    max_fh = max(kernels, key=lambda x: x.shape[0]).shape[0]
    max_fw = max(kernels, key=lambda x: x.shape[1]).shape[1]
    (hfh, hfw) = (max_fh-1)/2, (max_fw-1)/2
    img_bord = cv2.copyMakeBorder(image,hfh,hfh,hfw,hfw,cv2.BORDER_REFLECT_101)
    
    image_fd = np.fft.fft2(img_bord, axes=(-3,-2))
    (n, m, c) = img_bord.shape
    kernels_fd = np.array([np.fft.fft2(k, (n, m)) for k in kernels])
    # Make BGR kernels
    kernels_fd = np.repeat(kernels_fd[...,None], 3, axis=-1)
    
    Gx, Gy = compute_prior_L2_frequency(image_fd)
    outputs = deconvolution_L2_frequency(image_fd, kernels_fd, Gx, Gy, 0.01)
    
    return outputs