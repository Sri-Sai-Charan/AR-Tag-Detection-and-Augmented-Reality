#!/usr/bin/python3
import numpy as np
import cv2 as cv
from imagePreprocessing import *
from encoder import *

def main():
    cap = cv.VideoCapture('Media/1tagvideo.mp4')
    file = open("Media/kmatrix.xlsx - Sheet1.csv")
    K = np.loadtxt(file, delimiter=",")
    corners_projection = np.array([[0,0],[160,0],[160,160],[0,160]])

    (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')

    if int(major_ver)  < 3 :
        fps = cap.get(cv.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = cap.get(cv.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    vid = cv.VideoWriter('./Testudo.avi',cv.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if count>=0:
            if ret == False:
                break        
            
            fft_img = fft_func(frame)
            c_mask_1 = circular_mask_inner_and_outter(fft_img)
            ret, thres = cv.threshold(c_mask_1,100,255,cv.THRESH_BINARY)
            corners_outter = shi_tomasi_func(thres)
            if len(corners_outter) > 3 :
                c_mask_2 = circular_mask_outter(fft_img)
                outer_rem = remove_outter(c_mask_2,corners_outter)
                ret, thres_2 = cv.threshold(outer_rem,100,255,cv.THRESH_BINARY)
                corners_inner = shi_tomasi_func(thres_2)
                H = homography(corners_inner,corners_projection)
                gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
                ret ,  thres = cv.threshold(gray,127,255,cv.THRESH_BINARY)
                warp_world = warp_frame_to_camera(thres,H)
                search = True
                testudo = encoder(warp_world)
               
                if  testudo is None:
                    print("Reached")
                    search = False
                if search:
                    
                    warp_camera = warp_camera_to_frame(testudo,frame,H)
                    cv.namedWindow('final',cv.WINDOW_NORMAL)
                    cv.imshow("final",warp_camera)
                    vid.write(warp_camera)
                    
            else:
                vid.write(frame)
            cv.waitKey(1)
            
        print("Frame :",count)
        count+=1
    vid.release()
    cap.release()
if __name__ =='__main__':
    main()
