#!/usr/bin/python3
import numpy as np
import scipy.fftpack
from scipy.fftpack import fft, fft2, ifft
# from scipy import fft, fftfreq, fftshift
import cv2 as cv
import matplotlib.pyplot as plt
from imagePreprocessing import *
cv.namedWindow('frame',cv.WINDOW_NORMAL)
cv.resizeWindow('frame', (600, 600))


def main():
    cap = cv.VideoCapture('1tagvideo.mp4')
    file = open("kmatrix.xlsx - Sheet1.csv")
    K = np.loadtxt(file, delimiter=",")
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == False:
            break        
        # contour_mapping(frame)
        # blob_detection_func(frame)
        edges = canny_func(frame)
        # blob_function(frame)

        # gray = gray_conversion(frame)
        # fft_img = fft_func(gray)
        # shi_tomasi_func(edges)
        # edges=cv.cvtColor(edges,cv.COLOR_GRAY2BGR)
        # harris_func(edges)
        # break

        
        # break
    # cv.waitKey(0)
    cap.release()
if __name__ =='__main__':
    main()