#!/usr/bin/python3
import numpy as np
import scipy.fftpack
from scipy.fftpack import fft, fft2, ifft
# from scipy import fft, fftfreq, fftshift
import cv2 as cv
import matplotlib.pyplot as plt
cv.namedWindow('frame',cv.WINDOW_NORMAL)
cv.resizeWindow('frame', (600, 600))

# def find_edges(frame):
#     rows, cols = frame.shape
#     crow, ccol = int(rows / 2), int(cols / 2)  # center

#     # Circular HPF mask, center circle is 0, remaining all ones

#     mask = np.ones((rows, cols, 2), np.uint8)
#     r = 80
#     center = [crow, ccol]
#     x, y = np.ogrid[:rows, :cols]
#     mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
#     mask[mask_area] = 1

#     #calculating fft

#     dft = cv.dft(np.float32(frame), flags=cv.DFT_COMPLEX_OUTPUT)
#     dft_shift = np.fft.fftshift(dft)

#     magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))


#     # apply mask and inverse DFT
#     fshift = dft_shift * mask

#     fshift_mask_mag = 2000 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

#     f_ishift = np.fft.ifftshift(fshift)
#     img_back = cv.idft(f_ishift)
#     img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

#     plt.subplot(2, 2, 1), plt.imshow(frame, cmap='gray')
#     plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#     plt.subplot(2, 2, 2), plt.imshow(magnitude_spectrum, cmap='gray')
#     plt.title('After FFT'), plt.xticks([]), plt.yticks([])
#     plt.subplot(2, 2, 3), plt.imshow(fshift_mask_mag, cmap='gray')
#     plt.title('FFT + Mask'), plt.xticks([]), plt.yticks([])
#     plt.subplot(2, 2, 4), plt.imshow(img_back, cmap='gray')
#     plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])
#     plt.show()

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
    # cv.imshow('frame',img_back)
    return img_back

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

# def    
    
def gray_conversion(frame):
    gray_img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    return gray_img


def main():
    cap = cv.VideoCapture('1tagvideo.mp4')
    file = open("kmatrix.xlsx - Sheet1.csv")
    K = np.loadtxt(file, delimiter=",")
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == False:
            break
        
        
        contour_mapping(frame)
        
        break
    cv.waitKey(0)
    cap.release()
if __name__ =='__main__':
    main()