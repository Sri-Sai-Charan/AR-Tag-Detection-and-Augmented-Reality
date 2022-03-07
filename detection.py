#!/usr/bin/python3
import numpy as np
import cv2 as cv
from imagePreprocessing import *
# cv.namedWindow('frame',cv.WINDOW_NORMAL)
from encoder import *
def main():
    cap = cv.VideoCapture('1tagvideo.mp4')
    file = open("kmatrix.xlsx - Sheet1.csv")
    K = np.loadtxt(file, delimiter=",")
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == False:
            break        
        
        fft_img = fft_func(frame)
        c_mask_1 = circular_mask_inner_and_outter(fft_img)
        ret, thres = cv.threshold(c_mask_1,100,255,cv.THRESH_BINARY)
        corners_outter = shi_tomasi_func(thres)
        c_mask_2 = circular_mask_outter(fft_img)
        outer_rem = remove_outter(c_mask_2,corners_outter)
        ret, thres_2 = cv.threshold(outer_rem,100,255,cv.THRESH_BINARY)
        corners_inner = shi_tomasi_func(thres_2)
        H = homography(corners_inner,frame)
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        ret ,  thres = cv.threshold(gray,127,255,cv.THRESH_BINARY)
        warp = warp_frame_to_camera(thres,H)
        encoder(warp)

        # warp_camera_to_frame(warp,frame,H)
        # cv.imshow('frame',frame)
        # cv.waitKey(1)
        cv.waitKey(0)
        break
    cap.release()
if __name__ =='__main__':
    main()