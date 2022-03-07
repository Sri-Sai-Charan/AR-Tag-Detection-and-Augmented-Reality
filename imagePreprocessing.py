#!/usr/bin/python3 
from cv2 import GaussianBlur, cvtColor
import numpy as np
import scipy.fftpack
from scipy.fftpack import fft, fft2, ifft
import cv2 as cv
from matplotlib import pyplot as plt



def fft_func(frame):
    img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    ret , img = cv.threshold(img,127,255,cv.THRESH_BINARY)
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    
    dft_shift = np.fft.fftshift(dft)
    # magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 350
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    min_scale = np.min(img_back)
    max_scale = np.max(img_back)
    img_back = (img_back - min_scale)/max_scale
    img_back = np.uint8(img_back*255)

    # cv.imshow('frame',img_back)
    return img_back

def shi_tomasi_func(frame):
    corners = cv.goodFeaturesToTrack(frame,500,0.001,1)
    if corners is None:
        return [0]
    corners = np.int0(corners)
    x_arr = []
    y_arr  = []

    my_frame = frame.copy()
    for i in corners:
        x,y = i.ravel()
        x_arr.append(x)
        y_arr.append(y)
    #     cv.circle(frame,(x,y),3,(255,0,0),10)
    # cv.namedWindow("shi_tomasi all corners",cv.WINDOW_NORMAL)
    # cv.imshow("shi_tomasi all corners",frame)
    
    x_max = np.max(x_arr[:])
    y_max = np.max(y_arr[:])
    x_min = np.min(x_arr[:])
    y_min = np.min(y_arr[:])

    pt1 = np.argwhere(corners[:,0,0]==x_max) # left 
    pt2 = np.argwhere(corners[:,0,0]==x_min) # right 
    pt3 = np.argwhere(corners[:,0,1]==y_max) # top
    pt4 = np.argwhere(corners[:,0,1]==y_min) # bottom

    my_corners =[]
    my_corners.append([np.max(corners[pt4,0,0]),y_min]) # top left    
    my_corners.append([x_max,np.min(corners[pt1,0,1])]) # top right
    my_corners.append([np.max(corners[pt3,0,0]),y_max]) # bottom right
    my_corners.append([x_min,np.min(corners[pt2,0,1])]) # Bottom left
    
    # my_corners = rolling_avg(my_corners)
    # print(my_corners)
    # for i in my_corners:
    #     cv.circle(my_frame,(i[0],i[1]),3,(255,0,0),10)
    # cv.namedWindow("shi_tomasi",cv.WINDOW_NORMAL)
    # cv.imshow("shi_tomasi",my_frame)
    # print(my_corners)
    return np.array(my_corners)

array_for_avg = []

def rolling_avg(arr):
    global array_for_avg
    array_for_avg.append(arr)

    if len(array_for_avg) < 2:
        array_for_avg.append(arr)
        return arr
    else:
        array_for_avg.pop(0)
        array_for_avg.append(arr)
        return np.int32(np.mean(np.array(array_for_avg),axis=0))

def circular_mask_inner_and_outter(frame):
    rows, cols = frame.shape
    edges = cv.Canny(frame, threshold1=100, threshold2=200)
    
    avg_mat = np.argwhere(edges)

    ix = avg_mat[:,0]
    iy = avg_mat[:,1]
    ix_mean = np.mean(ix)
    iy_mean = np.mean(iy)
    center = [ix_mean,iy_mean]


    mask = np.zeros((rows, cols), np.uint8)
    r = 500
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 1
    frame = mask*frame
    r = 250
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    frame = mask*frame
    # cv.namedWindow('Circular mask', cv.WINDOW_NORMAL)
    # cv.imshow('Circular mask',frame)

    return frame

def homography(corners_1,corners_2):
    xw1,yw1 = corners_1[2]
    xw2,yw2 = corners_1[3]
    xw3,yw3 = corners_1[0]
    xw4,yw4 = corners_1[1]
    xc1, yc1 = corners_2[0]
    xc2 , yc2 = corners_2[1]
    xc3 , yc3 = corners_2[2]
    xc4 , yc4 = corners_2[3]
    corners2 = np.array([
        [xc1,yc1],
        [xc2,yc2],
        [xc3,yc3],
        [xc4,yc4]
    ])
    A = np.array([[xw1, yw1, 1, 0, 0, 0, -xc1*xw1, -xc1*yw1, -xc1],
         [0, 0, 0, xw1, yw1, 1, -yc1*xw1, -yc1*yw1, -yc1],
         [xw2, yw2, 1, 0, 0, 0, -xc2*xw2, -xc2*yw2, -xc2],
         [0, 0, 0, xw2, yw2, 1, -yc2*xw2, -yc2*yw2, -yc2],
         [xw3, yw3, 1, 0, 0, 0, -xc3*xw3, -xc3*yw3, -xc3],
         [0, 0, 0, xw3, yw3, 1, -yc3*xw3, -yc3*yw3, -yc3],
         [xw4, yw4, 1, 0, 0, 0, -xc4*xw4, -xc4*yw4, -xc4],
         [0, 0, 0, xw4, yw4, 1, -yc4*xw4, -yc4*yw4, -yc4]])
    
    m, n = A.shape

    AA_t = np.dot(A, A.transpose())
    A_tA = np.dot(A.transpose(), A) 

    eigen_values_1, U = np.linalg.eig(AA_t)
    eigen_values_2, V = np.linalg.eig(A_tA)
    index_1 = np.flip(np.argsort(eigen_values_1))
    eigen_values_1 = eigen_values_1[index_1]
    U = U[:, index_1]
    index_2 = np.flip(np.argsort(eigen_values_2))
    eigen_values_2 = eigen_values_2[index_2]
    V = V[:, index_2]

    E = np.zeros([m, n])

    var = np.minimum(m, n)

    for j in range(var):
        E[j,j] = np.abs(np.sqrt(eigen_values_1[j]))  

    Homography_Mat_ver = V[:, V.shape[1] - 1]
    Homography = Homography_Mat_ver.reshape([3,3])
    Homography = Homography / Homography[2,2]

    return Homography

def circular_mask_outter(frame):
    img = frame.copy()
    rows, cols = img.shape
    edges = cv.Canny(img, threshold1=100, threshold2=200)
    
    avg_mat = np.argwhere(edges)

    ix = avg_mat[:,0]
    iy = avg_mat[:,1]
    ix_mean = np.mean(ix)
    iy_mean = np.mean(iy)
    center = [ix_mean,iy_mean]


    mask = np.zeros((rows, cols), np.uint8)
    r = 500
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 1
    img = mask*img

    return img

def remove_outter(frame,corners):
    img = frame.copy()
    cv.polylines(img,[corners],True,(0,0,0),80)
    # cv.namedWindow("Outter Removed",cv.WINDOW_NORMAL)
    # cv.imshow("Outter Removed",img)
    return img

def warp_frame_to_camera(image, H):
    
    H_inv=np.linalg.inv(H)
    warped=np.zeros((160,160),np.uint8)
    for a in range(warped.shape[0]):
        for b in range(warped.shape[1]):
            f = [a,b,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            if (1080 > int(y/z) > 0) and (1920 > int(x/z) > 0):
                warped[a][b] = image[int(y/z)][int(x/z)]
    # cv.imshow('Warped',warped)
    return(warped)



def warp_camera_to_frame(testudo,frame,H):
    H_inv=np.linalg.inv(H)
    # unwarped= frame.copy()
    # print(testudo.shape)
    for a in range(testudo.shape[0]):
        for b in range(testudo.shape[0]):
            f = [a,b,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            y_dash = int(y / z)
            x_dash = int(x / z)
            if (1080 > y_dash > 0) and (1920 > x_dash > 0):
                 frame[y_dash][x_dash] =testudo[a][b] 
    # cv.namedWindow('unwarped',cv.WINDOW_NORMAL)
    # GaussianBlur =cv.GaussianBlur(frame,(5,5),0)
    # median = cv.medianBlur(GaussianBlur,5)
    # blur = cv.bilateralFilter(median,9,75,75)
    # cv.imshow('unwarped',blur)
    return(frame)