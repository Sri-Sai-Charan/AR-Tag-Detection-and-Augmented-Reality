import numpy as np
import cv2 as cv

def half_planes(my_map,point1,point2,color,upper):
    m = (point2[0]-point1[0])/(point2[1]-point1[1]+(1e-6))
    temp=np.zeros_like(my_map)
    for y  in range(0,80):
        c = point1[0] - m*point1[1]
        for x in range(0,80):
            if upper :
                if (y <= ((m*x)+c)):
                    temp[x,y]= color
            else:
                if (y >= ((m*x)+c)):
                    temp[x,y]= color
    return temp

def encoder(frame):
    img = frame.copy()
    img[:40,:] = 0
    img[120:,:] = 0
    img[:,:40] = 0
    img[:,120:] = 0

    x_arr = []
    y_arr  = []

    ones_location = np.argwhere(img[:,:] == 255)
    for i in ones_location:
        x,y = i.ravel()
        x_arr.append(x)
        y_arr.append(y)

    
    x_max = np.max(x_arr[:])
    y_max = np.max(y_arr[:])
    x_min = np.min(x_arr[:])
    y_min = np.min(y_arr[:])
    # print(x_min,x_max,y_min,y_max)
    # pt1 = np.argwhere(ones_location[:,0]==x_max) # left 
    # pt2 = np.argwhere(ones_location[:,0]==x_min) # right 
    # pt3 = np.argwhere(ones_location[:,1]==y_max) # top
    # pt4 = np.argwhere(ones_location[:,1]==y_min) # bottom

    # my_corners =[]
    # my_corners.append([np.max(ones_location[pt4,0]),y_min]) # top left    
    # my_corners.append([x_max,np.min(ones_location[pt1,1])]) # top right
    # my_corners.append([np.max(ones_location[pt3,0]),y_max]) # bottom right
    # my_corners.append([x_min,np.min(ones_location[pt2,1])]) # Bottom left
    # print(my_corners)
    encoder_img = np.zeros((80,80,3),dtype=np.int8)
    encoder_img = img[x_min:x_max,y_min:y_max]
    x_lower = np.int32((x_max - x_min)/4)
    x_upper = x_lower*3
    y_lower = np.int32((y_max-y_min)/4)
    y_upper = y_lower*3
    top_left_img = encoder_img[:x_lower,:y_lower]
    top_right_img = encoder_img[:x_lower,y_upper:]
    bottom_left_img = encoder_img[x_upper:,:y_lower]
    bottom_right_img = encoder_img[x_upper:,y_upper:]
    
    cv.namedWindow('encoder 1',cv.WINDOW_NORMAL)
    cv.namedWindow('top left',cv.WINDOW_NORMAL)
    cv.namedWindow('top right',cv.WINDOW_NORMAL)
    cv.namedWindow('bottom left',cv.WINDOW_NORMAL)
    cv.namedWindow('bottom right',cv.WINDOW_NORMAL)
    cv.imshow('encoder 1',encoder_img)
    cv.imshow('bottom left',bottom_left_img)
    cv.imshow('top left',top_left_img )
    cv.imshow('top right',top_right_img)
    cv.imshow('bottom right',bottom_right_img)
