# # %matplotlib inline
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.colors
# import numpy as np
# from PIL import Image
# # from skimage import data
# import scipy
# import math
# from scipy.ndimage import measurements
# # from skimage import data
# # from ipywidgets import interact, fixed, FloatSlider, IntSlider,FloatRangeSlider, Label


# def display_line_result(og_img, hough_img, edges):
    
#     current_image=og_img
#     w, h = current_image.shape
#     output_image = np.empty((w, h, 3))
#     edges = cv2.Canny(current_image,50,150,apertureSize =3)
#     output_image[:, :, 2] =  output_image[:, :, 1] =  output_image[:, :, 0] =  current_image/255.
#     lines = cv2.HoughLines(edges,1,np.pi/180,120)
#     max_size=max(w,h)**2
#     for rho_theta in lines:
#         rho=rho_theta[0][0]
#         theta=rho_theta[0][1]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + max_size*(-b))
#         y1 = int(y0 + max_size*(a))
#         x2 = int(x0 - max_size*(-b))
#         y2 = int(y0 - max_size*(a))
#         cv2.line(output_image,(x1,y1),(x2,y2),(1,0,0),1)
    
    
#     fig2, axes_array = plt.subplots(1, 4)
#     fig2.set_size_inches(9,3)
#     image_plot = axes_array[0].imshow(og_img,cmap=plt.cm.gray) 
#     axes_array[0].axis('off')
#     axes_array[0].set(title='Original Image')
#     image_plot = axes_array[1].imshow(edges,cmap=plt.cm.gray)
#     axes_array[1].axis('off')
#     axes_array[1].set(title='Edged Image')
#     image_plot = axes_array[2].imshow(hough_img)
#     axes_array[2].axis('off')
#     axes_array[2].set(title='Hough Lines Image')
#     image_plot = axes_array[3].imshow(output_image)
#     axes_array[3].axis('off')
#     axes_array[3].set(title='Hough Lines Open CV')
#     plt.show()
#     return

# def get_canny(og_img):
#     edged_image = cv2.Canny(og_img,50,150,apertureSize = 3)#current_image=data.checkerboard()
#     return edged_image

# def hough_lines(og_img,rho_resolution,theta_resolution,threshold,edges):
#     rho_theta_values = []
#     width, height = og_img.shape
#     hough_img = np.empty((width, height, 3))
#     hough_img[:, :, 2] =  hough_img[:, :, 1] =  hough_img[:, :, 0] =  og_img/255.
    
#     digonal = math.sqrt(width*width + height*height)
#     max_size=max(width,height)**2
    
#     thetas = np.linspace(0,180,theta_resolution+1)
#     rhos = np.linspace(-digonal,digonal,rho_resolution+1)
   
#     acc = np.zeros((rho_resolution+1,theta_resolution+1))

#     for x_index in range(0, width):
#         for y_index in range(0, height):
#             if edges[x_index][y_index] > 0:
#                 for t_index in range(0, len(thetas)):
#                     rho = x_index * math.cos(thetas[t_index]) + y_index * math.sin(thetas[t_index])
#                     for r_index in range(0, len(rhos)):
#                         if rhos[r_index]>rho:
#                             break
#                     acc[r_index][t_index] += 1
   
#     for rho_value in range(0, len(rhos)):
#         for t_value in range(0, len(thetas)):
#             if acc[rho_value][t_value] >= threshold:
#                 rho_theta_values.append([rhos[rho_value], thetas[t_value]])

    
#     for rho_theta in rho_theta_values:
#         rho=rho_theta[0]
#         theta=rho_theta[1]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + max_size*(-b))
#         y1 = int(y0 + max_size*(a))
#         x2 = int(x0 - max_size*(-b))
#         y2 = int(y0 - max_size*(a))
#         cv2.line(hough_img,(x1,y1),(x2,y2),(1,0,0),1)
    
#     display_line_result(og_img, hough_img, edges)
#     return
    
# def hough_transform(frame):
#     og_img = frame
#     edges = get_canny(og_img)   

#     hough_lines(og_img = og_img,
#             rho_resolution=150,
#             theta_resolution=360,
#             threshold=180,
#             edges= edges)
    # interact(hough_lines,
    #         og_img = fixed(og_img),
    #         rho_resolution=IntSlider(min=10, max=1000, step=1,value=150,continuous_update=False),
    #         theta_resolution=IntSlider(min=10, max=1000, step=1,value=360,continuous_update=False),
    #         threshold=IntSlider(min=5, max=1000, step=1,value=180,continuous_update=False),
    #         edges= edges) 

import numpy as np
import imageio 
import math

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    """
    Hough transform for lines
    Input:
    img - 2D binary image with nonzeros representing edges
    angle_step - Spacing between angles to use every n-th angle
                 between -90 and 90 degrees. Default step is 1.
    lines_are_white - boolean indicating whether lines to be detected are white
    value_threshold - Pixel values above or below the value_threshold are edges
    Returns:
    accumulator - 2D array of the hough transform accumulator
    theta - array of angles used in computation, in radians.
    rhos - array of rho values. Max size is 2 times the diagonal
           distance of the input image.
    """
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos

def fast_hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    """hough line using vectorized numpy operations,
    may take more memory, but takes much less time"""
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step)) #can be changed
    #width, height = col.size  #if we use pillow
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    #are_edges = cv2.Canny(img,50,150,apertureSize = 3)
    y_idxs, x_idxs = np.nonzero(are_edges)  # (row, col) indexes to edges
    # Vote in the hough accumulator
    xcosthetas = np.dot(x_idxs.reshape((-1,1)), cos_theta.reshape((1,-1)))
    ysinthetas = np.dot(y_idxs.reshape((-1,1)), sin_theta.reshape((1,-1)))
    rhosmat = np.round(xcosthetas + ysinthetas) + diag_len
    rhosmat = rhosmat.astype(np.int16)
    for i in xrange(num_thetas):
        rhos,counts = np.unique(rhosmat[:,i], return_counts=True)
        accumulator[rhos,i] = counts
    return accumulator, thetas, rhos

def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    # plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    imgpath = 'imgs/binary_crosses.png'
    img = imageio.imread(imgpath)
    if img.ndim == 3:
        img = rgb2gray(img)
    accumulator, thetas, rhos = hough_line(img)
    show_hough_line(img, accumulator, thetas, rhos, save_path='imgs/output.png')