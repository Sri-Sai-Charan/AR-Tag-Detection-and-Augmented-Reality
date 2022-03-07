import numpy as np
import cv2 as cv

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

        if len(x_arr) == 0 or len(y_arr)==0:
            return None

        x_max = np.max(x_arr[:])
        y_max = np.max(y_arr[:])
        x_min = np.min(x_arr[:])
        y_min = np.min(y_arr[:])

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

        inner_top_left = encoder_img[x_lower:x_lower*2,  y_lower:y_lower*2]
        inner_top_right = encoder_img[x_lower*2:x_lower*3, y_lower:y_lower*2]
        inner_bottom_left = encoder_img[x_lower:x_lower*2, y_lower*2:y_lower*3]
        inner_bottom_right = encoder_img[x_lower*2:x_lower*3, y_lower*2:y_lower*3]
        bit_encoder_img = np.array([inner_top_left,inner_top_right,inner_bottom_right,inner_bottom_left])
        bit_encoder = [1,1,1,1]

        for i in range(4):
            number_of_zeros = np.argwhere(bit_encoder_img[i] == 0)
            if len(number_of_zeros) > 15:
                bit_encoder[i] = 0

        ############################################
        #UNCOMMENT TO VISUALIZE ENCODER
        ############################################
        # cv.namedWindow('encoder 1',cv.WINDOW_NORMAL)
        # cv.namedWindow('top left',cv.WINDOW_NORMAL)
        # cv.namedWindow('top right',cv.WINDOW_NORMAL)
        # cv.namedWindow('bottom left',cv.WINDOW_NORMAL)
        # cv.namedWindow('bottom right',cv.WINDOW_NORMAL)
        # cv.imshow('encoder 1',encoder_img)
        # cv.imshow('bottom left',bottom_left_img)
        # cv.imshow('top left',top_left_img )
        # cv.imshow('top right',top_right_img)
        # cv.imshow('bottom right',bottom_right_img)

        testudo_img = cv.imread('Media/testudo.png')

        resized_testudo = cv.resize(testudo_img,(160,160))

        top_left_flag = np.argwhere(top_left_img==0)

        if len(top_left_flag) < 30:
            rotated_testudo = rotate_img(resized_testudo,-90)
            bits_inorder = bit_encoder
            val =caluculate_bits(bits_inorder)
            print("Tl",val)
            return rotated_testudo

        top_right_flag = np.argwhere(top_right_img==0)

        if len(top_right_flag) < 30 :
            rotated_testudo = rotate_img(resized_testudo,180)
            bits_inorder = [bit_encoder[1],bit_encoder[2],bit_encoder[3],bit_encoder[0]]
            val = caluculate_bits(bits_inorder)
            print("Tr",val)
            return rotated_testudo

        bottom_left_flag = np.argwhere(bottom_left_img==0)

        if len(bottom_left_flag) < 30 :
            rotated_testudo = rotate_img(resized_testudo,0)
            bits_inorder = [bit_encoder[2],bit_encoder[3],bit_encoder[0],bit_encoder[1]]
            val =caluculate_bits(bits_inorder)
            print("Bl",val)
            return rotated_testudo
 
        bottom_right_flag = np.argwhere(bottom_right_img==0)

        if len(bottom_right_flag) < 30 :
            bits_inorder = [bit_encoder[3],bit_encoder[0],bit_encoder[1],bit_encoder[2]]
            val = caluculate_bits(bits_inorder)
            rotated_testudo = rotate_img(resized_testudo,90)
            print("Br",val)
            return rotated_testudo
        
        rotated_testudo = rotate_img(resized_testudo,0)
        return rotated_testudo

def caluculate_bits(bit_array):
    val=bit_array[0]
    val = val + (bit_array[1]*2)
    val = val + (bit_array[2]*4)
    val = val + (bit_array[3]*8)
    return val

    
    
def rotate_img(frame,deg):
    image = frame.copy()
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv.getRotationMatrix2D((cX, cY), deg, 1.0)
    rotated = cv.warpAffine(image, M, (w, h))

    return rotated

