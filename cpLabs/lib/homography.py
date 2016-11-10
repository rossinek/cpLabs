import numpy as np
import cv2

def calculate_coords(y, x, H):
    old_coordinates = [
        [y],
        [x],
        [1]
    ]
    
    [newY, newX, newW] = np.dot(H, old_coordinates)
    return (newY / newW, newX / newW)

def between(v, min_v, max_v):
    return v>=min_v and v<max_v

def apply_homography_nn(train_img, poster_img, H):
    output_img = train_img.copy()

    (height, width, channels) = output_img.shape
    (poster_h, poster_w, poster_c) = poster_img.shape

    for y in range(0, height):
        for x in range(0, width):
            (y_pos, x_pos) = calculate_coords(y, x, H)
            y_pos = int(y_pos)
            x_pos = int(x_pos)

            if between(y_pos, 0, poster_h) and between(x_pos, 0, poster_w):
                output_img[y, x] = poster_img[y_pos, x_pos]
    return output_img

def bilinear_interpolation(y,x,y1,y2,x1,x2,v11,v12,v21,v22):
    denominator = float((x2-x1)*(y2-y1))
    result  = (((x2-x)*(y2-y))/denominator)*v11
    result += (((x-x1)*(y2-y))/denominator)*v12
    result += (((x2-x)*(y-y1))/denominator)*v21
    result += (((x-x1)*(y-y1))/denominator)*v22
    return int(result)

def apply_homography_bi(train_img, poster_img, H):
    output_img = train_img.copy()

    (height, width, channels) = output_img.shape
    (poster_h, poster_w, poster_c) = poster_img.shape

    for y in range(0, height):
        for x in range(0, width):
            (y_pos, x_pos) = calculate_coords(y, x, H)
            y1 = int(y_pos)
            y2 = y1+1
            x1 = int(x_pos)
            x2 = x1+1

            if  between(y1, 0, poster_h) \
            and between(x1, 0, poster_w) \
            and between(y2, 0, poster_h) \
            and between(x2, 0, poster_w):
                for i in range(0, channels):
                    output_img[y, x, i] = bilinear_interpolation(
                        y_pos,x_pos,
                        y1,y2,x1,x2,
                        poster_img[y1, x1, i],
                        poster_img[y1, x2, i],
                        poster_img[y2, x1, i],
                        poster_img[y2, x2, i]
                    )
    return output_img

def apply_homography_bi2(train_img, poster_img, H):
    output_img = train_img.copy()

    (height, width, channels) = output_img.shape
    (poster_h, poster_w, poster_c) = poster_img.shape

    for y in range(0, height):
        for x in range(0, width):
            (y_pos, x_pos) = calculate_coords(y, x, H)
            y1 = int(y_pos)
            y2 = y1+1
            x1 = int(x_pos)
            x2 = x1+1

            by1 = between(y1, 0, poster_h)
            bx1 = between(x1, 0, poster_w)
            by2 = between(y2, 0, poster_h)
            bx2 = between(x2, 0, poster_w)
            if by1 or bx1 or by2 or bx2:
                for i in range(0, channels):
                    v11=v12=v21=v22 = output_img[y, x, i]
                    if by1 and bx1:
                        v11 = poster_img[y1, x1, i]
                    if by1 and bx2:
                        v12 = poster_img[y1, x2, i]
                    if by2 and bx1:
                        v21 = poster_img[y2, x1, i]
                    if by2 and bx2:
                        v22 = poster_img[y2, x2, i]

                    output_img[y, x, i] = bilinear_interpolation(
                        y_pos,x_pos,
                        y1,y2,x1,x2,
                        v11,v12,v21,v22
                    )
    return output_img

def apply_homography_bi3(img1, img2, H, output_shape=None, offset=(0,0)):
    if(output_shape):
        output_img = np.zeros(output_shape, img1.dtype)
    else:
        output_img = np.zeros_like(img1)

    (height, width, channels) = output_img.shape
    (h2, w2, c2) = img2.shape
    (h1, w1, c1) = img1.shape
    (offset_y, offset_x) = offset
    
    for y in range(0, height):
        for x in range(0, width):
            (y_pos, x_pos) = calculate_coords(y-offset_y, x-offset_x, H)
            y1 = int(y_pos)
            y2 = y1+1
            x1 = int(x_pos)
            x2 = x1+1

            if  between(y1, 0, h2) \
            and between(x1, 0, w2) \
            and between(y2, 0, h2) \
            and between(x2, 0, w2):
                for i in range(0, channels):
                    output_img[y, x, i] = bilinear_interpolation(
                        y_pos,x_pos,
                        y1,y2,x1,x2,
                        img2[y1, x1, i],
                        img2[y1, x2, i],
                        img2[y2, x1, i],
                        img2[y2, x2, i]
                    )
            else:
                if between(y-offset_y, 0, h1) \
                and between(x-offset_x, 0, w1):
                    output_img[y, x] = img1[y-offset_y, x-offset_x]
    return output_img

def find_homography(p1, p2):
    A = np.zeros((8, 9), np.float32)
    for i in range(0,4): 
        A[2*i]  = [-p1[i,0],-p1[i,1],-p1[i,2],0,0,0,(p2[i,0]/p2[i,2])*p1[i,0],(p2[i,0]/p2[i,2])*p1[i,1],(p2[i,0]/p2[i,2])*p1[i,2]]
        A[2*i+1]= [0,0,0,-p1[i,0],-p1[i,1],-p1[i,2],(p2[i,1]/p2[i,2])*p1[i,0],(p2[i,1]/p2[i,2])*p1[i,1],(p2[i,1]/p2[i,2])*p1[i,2]]

    U, s, V = np.linalg.svd(A, full_matrices=True)
    H = V[8,:]
    return H.reshape((3,3))

def find_bounding_box((h1,w1,c1), (h2,w2,c2), H):
    X = [0,w1]
    Y = [0,h1]
    H_inv = np.linalg.inv(H)
    corners = [(0,0), (0,w2), (h2,0), (h2,w2)]
    for (y,x) in corners:
        (new_y, new_x) = calculate_coords(y,x,H_inv)
        X.append(int(new_x))
        Y.append(int(new_y))
    
    new_shape = (max(Y)-min(Y), max(X)-min(X), c1)
    offset = (0-min(Y), 0-min(X))
    
    return new_shape, offset


def bilinear_interpolation_float(y,x,y1,y2,x1,x2,v11,v12,v21,v22):
    denominator = float((x2-x1)*(y2-y1))
    result  = (((x2-x)*(y2-y))/denominator)*v11
    result += (((x-x1)*(y2-y))/denominator)*v12
    result += (((x2-x)*(y-y1))/denominator)*v21
    result += (((x-x1)*(y-y1))/denominator)*v22
    return result