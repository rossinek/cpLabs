import cv2
import numpy as np

def main():
	train_img = cv2.imread("cpLabs/lab3/images/green.png")
	poster_img = cv2.imread("cpLabs/lab3/images/poster.png")
	(ph,pw,pc) = poster_img.shape
	
	poster_p = np.array([[0, 0, 1], [0, pw, 1], [ph, pw, 1], [ph, 0, 1]])
	train_p = np.array([[170, 95, 1], [171,238, 1], [233,235, 1], [239,94, 1]])

	#H = np.linalg.inv(find_homography(train_p,poster_p))
	H = find_homography(train_p,poster_p)
	print H
	output_img = apply_homography_nn(train_img, poster_img, H)

	cv2.imshow("Image", output_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def find_homography(p1, p2):
	A = np.zeros((8, 9), np.float32)
	for i in range(0,4): 
		A[2*i] 	= [-p1[i,0],-p1[i,1],-p1[i,2],0,0,0,(p2[i,0]/p2[i,2])*p1[i,0],(p2[i,0]/p2[i,2])*p1[i,1],(p2[i,0]/p2[i,2])*p1[i,2]]
		A[2*i+1]= [0,0,0,-p1[i,0],-p1[i,1],-p1[i,2],(p2[i,1]/p2[i,2])*p1[i,0],(p2[i,1]/p2[i,2])*p1[i,1],(p2[i,1]/p2[i,2])*p1[i,2]]

	U, s, V = np.linalg.svd(A, full_matrices=True)
	H = V[8,:]
	return H.reshape((3,3))

def apply_homography_nn(train_img, poster_img, H):
    output_img = train_img.copy()

    (height, width, channels) = output_img.shape
    (poster_h, poster_w, poster_c) = poster_img.shape

    for y in range(0, height):
        for x in range(0, width):
            (y_pos, x_pos) = calculate_coords(y, x, H)
            y_pos = int(y_pos)
            x_pos = int(x_pos)

            if __between(y_pos, 0, poster_h) and __between(x_pos, 0, poster_w):
                output_img[y, x] = poster_img[y_pos, x_pos]
    return output_img

def calculate_coords(y, x, H):
    old_coordinates = [
        [y],
        [x],
        [1]
    ]
    
    [newY, newX, newW] = np.dot(H, old_coordinates)
    return (newY / newW, newX / newW)

def __between(v, min_v, max_v):
    return v>=min_v and v<max_v

main()