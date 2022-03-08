#!/usr/bin/python3
import numpy as np
import cv2 as cv
from imagePreprocessing import homography
from encoder import *

def fft_func(frame):
    img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    ret , img = cv.threshold(img,150,255,cv.THRESH_BINARY)
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 330
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
    edges = cv.Canny(img_back, threshold1=100, threshold2=200)

    ret , img_back = cv.threshold(edges,10,255,cv.THRESH_BINARY)
    return img_back

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
    r = 200
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    frame = mask*frame


    return frame

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

    return img



def shi_tomasi_func(frame):
    corners = cv.goodFeaturesToTrack(frame,500,0.0001,0.5)
    if corners is None:
        return [0]
    corners = np.int0(corners)
    x_arr = []
    y_arr  = []

    for i in corners:
        x,y = i.ravel()
        x_arr.append(x)
        y_arr.append(y)

    
    x_max = np.max(x_arr[:])
    y_max = np.max(y_arr[:])
    x_min = np.min(x_arr[:])
    y_min = np.min(y_arr[:])

    pt1 = np.argwhere(corners[:,0,0]==x_max)
    pt2 = np.argwhere(corners[:,0,0]==x_min) 
    pt3 = np.argwhere(corners[:,0,1]==y_max)
    pt4 = np.argwhere(corners[:,0,1]==y_min) 

    my_corners =[]
    my_corners.append([np.max(corners[pt4,0,0]),y_min]) # top left    
    my_corners.append([x_max,np.min(corners[pt1,0,1])]) # top right
    my_corners.append([np.max(corners[pt3,0,0]),y_max]) # bottom right
    my_corners.append([x_min,np.min(corners[pt2,0,1])]) # Bottom left
 
    return np.array(my_corners)

def projectionMatrix(H, K):  
    h1 = H[:,0]          
    h2 = H[:,1]
    h3 = H[:,2]
    l = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K),h2)))
    b_t = l * np.matmul(np.linalg.inv(K),H)
    det = np.linalg.det(b_t)

    if det > 0:
        b = b_t
    else:                   
        b = -1 * b_t  
        
    r_1 = b[:, 0]
    r_2 = b[:, 1]                      
    r_3 = np.cross(r_1, r_2)
    t = b[:, 2]
    Rt = np.column_stack((r_1, r_2, r_3, t))
    P = np.matmul(K,Rt)  
    return(P,Rt,t)

def plot_cube_lines(P,frame):

    x_1,y_1,z_1 = np.matmul(P,[0,0,0,1])
    x_2,y_2,z_2 = np.matmul(P,[0,159,0,1])
    x_3,y_3,z_3 = np.matmul(P,[159,0,0,1])
    x_4,y_4,z_4 = np.matmul(P,[159,159,0,1])
    x_5,y_5,z_5 = np.matmul(P,[0,0,-159,1])
    x_6,y_6,z_6 = np.matmul(P,[0,159,-159,1])
    x_7,y_7,z_7 = np.matmul(P,[159,0,-159,1])
    x_8,y_8,z_8 = np.matmul(P,[159,159,-159,1])
    
    #Bottom Square
    cv.line(frame,(int(x_1/z_1),int(y_1/z_1)),(int(x_5/z_5),int(y_5/z_5)), (255,0,0), 10)
    cv.line(frame,(int(x_2/z_2),int(y_2/z_2)),(int(x_6/z_6),int(y_6/z_6)), (255,0,0), 10)
    cv.line(frame,(int(x_3/z_3),int(y_3/z_3)),(int(x_7/z_7),int(y_7/z_7)), (255,0,0), 10)
    cv.line(frame,(int(x_4/z_4),int(y_4/z_4)),(int(x_8/z_8),int(y_8/z_8)), (255,0,0), 10)
    #Lines connecting top and bottom
    cv.line(frame,(int(x_1/z_1),int(y_1/z_1)),(int(x_2/z_2),int(y_2/z_2)), (0,255,0), 10)
    cv.line(frame,(int(x_1/z_1),int(y_1/z_1)),(int(x_3/z_3),int(y_3/z_3)), (0,255,0), 10)
    cv.line(frame,(int(x_2/z_2),int(y_2/z_2)),(int(x_4/z_4),int(y_4/z_4)), (0,255,0), 10)
    cv.line(frame,(int(x_3/z_3),int(y_3/z_3)),(int(x_4/z_4),int(y_4/z_4)), (0,255,0), 10)
    #Top Square
    cv.line(frame,(int(x_5/z_5),int(y_5/z_5)),(int(x_6/z_6),int(y_6/z_6)), (0,0,255), 10)
    cv.line(frame,(int(x_5/z_5),int(y_5/z_5)),(int(x_7/z_7),int(y_7/z_7)), (0,0,255), 10)
    cv.line(frame,(int(x_6/z_6),int(y_6/z_6)),(int(x_8/z_8),int(y_8/z_8)), (0,0,255), 10)
    cv.line(frame,(int(x_7/z_7),int(y_7/z_7)),(int(x_8/z_8),int(y_8/z_8)), (0,0,255), 10)

    return frame

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
    vid = cv.VideoWriter('./Cube_output.avi',cv.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
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
            if len(corners_outter) >3 :
                c_mask_2 = circular_mask_outter(fft_img)
                outer_rem = remove_outter(c_mask_2,corners_outter)
                ret, thres_2 = cv.threshold(outer_rem,100,255,cv.THRESH_BINARY)
                corners_inner = shi_tomasi_func(thres_2)
                
                H = homography(corners_projection,corners_inner)
                P,Rt,t = projectionMatrix(H,K)
                frame = plot_cube_lines(P,frame)
                vid.write(frame)
                cv.namedWindow('cube output',cv.WINDOW_NORMAL)
                cv.imshow('cube output',frame)
            else:
                vid.write(frame)
            cv.waitKey(1)
        print("Frame :",count)
        count+=1
    vid.release()
    cap.release()
 
if __name__ =='__main__':
    main()