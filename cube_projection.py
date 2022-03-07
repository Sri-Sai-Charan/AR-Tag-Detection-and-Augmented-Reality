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
    # magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
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
    # cv.namedWindow('fft',cv.WINDOW_NORMAL)
    # cv.imshow('fft',img_back)
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
    # cv.namedWindow('Circular mask', cv.WINDOW_NORMAL)
    # cv.imshow('Circular mask',frame)

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
    # cv.namedWindow("Outter Removed",cv.WINDOW_NORMAL)
    # cv.imshow("Outter Removed",img)
    return img



def shi_tomasi_func(frame):
    corners = cv.goodFeaturesToTrack(frame,500,0.0001,0.5)
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

def projectionMatrix(h, K):  
    h1 = h[:,0]          #taking column vectors h1,h2 and h3
    h2 = h[:,1]
    h3 = h[:,2]
    #calculating lamda
    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K),h2)))
    b_t = lamda * np.matmul(np.linalg.inv(K),h)

    #check if determinant is greater than 0 ie. has a positive determinant when object is in front of camera
    det = np.linalg.det(b_t)

    if det > 0:
        b = b_t
    else:                    #else make it positive
        b = -1 * b_t  
        
    row1 = b[:, 0]
    row2 = b[:, 1]                      #extract rotation and translation vectors
    row3 = np.cross(row1, row2)
    
    t = b[:, 2]
    Rt = np.column_stack((row1, row2, row3, t))
#     r = np.column_stack((row1, row2, row3))
    P = np.matmul(K,Rt)  
    return(P,Rt,t)




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
    # vid = cv.VideoWriter('./Cube_output.avi',cv.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
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
                
                #Determine the coordinates of the cube by multiplying with the projection matix
                H_cube = homography(corners_projection,corners_inner)
                P,Rt,t = projectionMatrix(H_cube,K)
                x1,y1,z1 = np.matmul(P,[0,0,0,1])
                x2,y2,z2 = np.matmul(P,[0,159,0,1])
                x3,y3,z3 = np.matmul(P,[159,0,0,1])
                x4,y4,z4 = np.matmul(P,[159,159,0,1])
                x5,y5,z5 = np.matmul(P,[0,0,-159,1])
                x6,y6,z6 = np.matmul(P,[0,159,-159,1])
                x7,y7,z7 = np.matmul(P,[159,0,-159,1])
                x8,y8,z8 = np.matmul(P,[159,159,-159,1])
                
                #Join the coordinates by using cv2.line function. Also divide by z to normalize
                
                cv.line(frame,(int(x1/z1),int(y1/z1)),(int(x5/z5),int(y5/z5)), (255,0,0), 2)
                cv.line(frame,(int(x2/z2),int(y2/z2)),(int(x6/z6),int(y6/z6)), (255,0,0), 2)
                cv.line(frame,(int(x3/z3),int(y3/z3)),(int(x7/z7),int(y7/z7)), (255,0,0), 2)
                cv.line(frame,(int(x4/z4),int(y4/z4)),(int(x8/z8),int(y8/z8)), (255,0,0), 2)

                cv.line(frame,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (0,255,0), 2)
                cv.line(frame,(int(x1/z1),int(y1/z1)),(int(x3/z3),int(y3/z3)), (0,255,0), 2)
                cv.line(frame,(int(x2/z2),int(y2/z2)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)
                cv.line(frame,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)

                cv.line(frame,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (0,0,255), 2)
                cv.line(frame,(int(x5/z5),int(y5/z5)),(int(x7/z7),int(y7/z7)), (0,0,255), 2)
                cv.line(frame,(int(x6/z6),int(y6/z6)),(int(x8/z8),int(y8/z8)), (0,0,255), 2)
                cv.line(frame,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (0,0,255), 2)
                cv.namedWindow("CUBE VIDEO",cv.WINDOW_NORMAL)
                cv.imshow("CUBE VIDEO", frame)
                # vid.write(frame)
                # cv.namedWindow('frame',cv.WINDOW_NORMAL)
                # cv.imshow('frame',frame)
            # else:
                # vid.write(frame)
            cv.waitKey(1)
            # cv.waitKey(0)
            # break
        print("Frame :",count)
        count+=1
    # vid.release()
    cap.release()
 
if __name__ =='__main__':
    main()