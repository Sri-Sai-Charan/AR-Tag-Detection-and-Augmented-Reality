#!/usr/bin/python3 
from cv2 import cvtColor
import numpy as np
import scipy.fftpack
from scipy.fftpack import fft, fft2, ifft
import cv2 as cv
from matplotlib import pyplot as plt
# cv.namedWindow('frame',cv.WINDOW_NORMAL)
# cv.resizeWindow('frame', (600, 600))

def fft_func(frame):
    img = frame
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.ones((rows, cols, 2), np.uint8)
    r = 80
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    fig = plt.figure()

    # ax1 = fig.add_subplot(2,2,1)
    # ax1.imshow(img, cmap='gray')
    # ax1.title.set_text('Input Image')
    # ax2 = fig.add_subplot(2,2,2)
    # ax2.imshow(magnitude_spectrum, cmap='gray')
    # ax2.title.set_text('FFT of image')
    # ax3 = fig.add_subplot(2,2,3)
    # ax3.imshow(fshift_mask_mag, cmap='gray')
    # ax3.title.set_text('FFT + Mask')
    # ax4 = fig.add_subplot(2,2,4)
    # fig.imshow(img_back, cmap='gray')
    # plt.imshow(img_back)
    # ax4.title.set_text('After inverse FFT')
    
    # plt.show()
    # print(img_back.shape)
    cv.imshow('frame',img_back)
    cv.waitKey(0)
    return img_back

    # plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 2), plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.title('After FFT'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 3), plt.imshow(fshift_mask_mag, cmap='gray')
    # plt.title('FFT + Mask'), plt.xticks([]), plt.yticks([])
    # plt.imshow(img_back, cmap='gray')
    # plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])
    # plt.show()
    

def shi_tomasi_func(frame):
    corners = cv.goodFeaturesToTrack(frame,25,0.01,10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv.circle(frame,(x,y),3,255,2)
    plt.imshow(frame, cmap='gray'),plt.show()

def corner_detection(frame):
    # gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(frame,25,0.01,10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv.circle(frame,(x,y),3,255,-1)
    plt.imshow(frame),plt.show()

def contour_mapping(frame):
    img_blur = cv.GaussianBlur(frame,(3,3), 0)
    edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200)
    img = edges 
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    i = 0
    out = np.zeros_like(img)
    for c in contours:
        if (30000 < int(cv.contourArea(c)) < 40000):           
            mask = np.zeros_like(img)
            cv.drawContours(mask,contours,i,(255),cv.FILLED) 
            # img_pl = np.zeros_like(img)/
            out[mask == 255] = img[mask == 255]
            # corner_detection(out)
            cv.imshow('frame',out)
            cv.waitKey(0)
        i+=1

center_array= []

def canny_func(frame):
    img_blur = cv.GaussianBlur(frame,(3,3), 0)

    # median = cv.medianBlur(img_blur, 5)
    gray = gray_conversion(frame)
    rows, cols = gray.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols), np.uint8)
    r = 450
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 1
    edges = cv.Canny(img_blur, threshold1=100, threshold2=200)
    edges = edges*mask
    blank = np.zeros_like(edges)
    
    avg_mat = np.argwhere(edges)
    # print(avg_mat.shape)



    ix = avg_mat[:,0]
    iy = avg_mat[:,1]
    ix_mean = np.mean(ix)
    iy_mean = np.mean(iy)
    # moving average
    # if(len(center_array)>=10):
    #     center_array.reverse()
    #     center_array.pop()
    #     center_array.reverse()


    #     ix_mean = np.mean(center_array[:][0][0])
    #     iy_mean = np.mean(center_array[:][0][1])
    #     print(len(center_array[0]))
    #     print("center",(ix_mean,iy_mean))
    center = [ix_mean,iy_mean]
    # print(center_array)
    # center_array.append(center)


    r= 200
    x, y = np.ogrid[:rows, :cols]
    mask_area2 = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask2 = np.zeros((rows, cols), np.uint8)
    mask2[mask_area2] = 1
    edges = edges*mask2
    cv.circle(edges,(ccol,crow),450,255,5)
    cv.circle(edges,(int(iy_mean),int(ix_mean)),200,255,3)
    cv.imshow('frame',edges)
    # cv.imshow('frame',blank)

    # if(cv.waitKey(0)==ord('q')):
    #     cv.destroyAllWindows()
    #     return
    cv.waitKey(0)
    return edges


# def blob_detection_func(frame):
#     ori = frame
#     im = gray_conversion(ori)
#     detector = cv.SimpleBlobDetector_create()
#     keypoints = detector.detect(im)
#     im_with_keypoints = cv.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     cv.namedWindow('Original', cv.WINDOW_NORMAL)
#     cv.namedWindow('BLOB', cv.WINDOW_NORMAL)
#     cv.imshow('Original',ori) 
#     cv.imshow('BLOB',im_with_keypoints)
#     if cv.waitKey(0) & 0xff == 27:
#         cv.destroyAllWindows()

def harris_func(frame):
    img = frame
    # gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # gray = np.float32(gray)
    dst = cv.cornerHarris(img,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[255]
    cv.namedWindow('dst',cv.WINDOW_NORMAL)
    cv.imshow('dst',img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

def gray_conversion(frame):
    median = cv.medianBlur(frame, 5)
    gray_img = cv.cvtColor(median,cv.COLOR_BGR2GRAY)
    return gray_img

def blob_function(frame):
    img = frame
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    detector = cv.SimpleBlobDetector(gray)
    blank = np.zeros((1,1))
    keypoints = detector.detect(img)
    blobs = cv.drawKeypoints(img, keypoints, blank, (0,255,255), cv.DRAW_MATCHES_FLAGS_DEFAULT)
    cv.namedWindow('Blob',cv.WINDOW_NORMAL)
    cv.imshow('Blob',blobs)

# def homography_func(i,w):

#     a = [[xw1, yw1, 1, 0, 0, 0, -xc1*xw1, -xc1*yw1, -xc1],
#          [0, 0, 0, xw1, yw1, 1, -yc1*xw1, -yc1*yw1, -yc1],
#          [xw2, yw2, 1, 0, 0, 0, -xc2*xw2, -xc2*yw2, -xc2],
#          [0, 0, 0, xw2, yw2, 1, -yc2*xw2, -yc2*yw2, -yc2],
#          [xw3, yw3, 1, 0, 0, 0, -xc3*xw3, -xc3*yw3, -xc3],
#          [0, 0, 0, xw3, yw3, 1, -yc3*xw3, -yc3*yw3, -yc3],
#          [xw4, yw4, 1, 0, 0, 0, -xc4*xw4, -xc4*yw4, -xc4],
#          [0, 0, 0, xw4, yw4, 1, -yc4*xw4, -yc4*yw4, -yc4]]