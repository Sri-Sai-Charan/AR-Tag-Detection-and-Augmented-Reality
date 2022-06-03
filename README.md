# AR-Tag-Detection-and-Augment-Reality
An individual project aimed at implementing Homography warpPerspective and findContours without using inbuilt OpenCV functions, to superimpose an image of our mascot, Testudo. Furthermore code has been implemented to superimpose a cube using camera's intrinsic values to map the cube onto a plane in world frame.

## Description

This code uses uses basic OpenCV function to carry out various masks and implement edge and corner detection without the use of findContours. The fast fourier transform is first used to filter out low frequency noise using a high-pass circular filter. Then the image is passed through a series of mask to extract the exact corners of the April Tag that can be seen in the sample video. After which we find the Homography from the four detected corners to preset corners to decode the April Tag to fetch data about its ID and its Orientation. This is accomplished throught the use of python sclies and numpy's funciton of max and min to determine how to dynamically set the upper and lower bounds of the warped April Tag. Once decoding is done we can rotate the given sample testudo image (to be superimposed onto the world frame). We can again use inverse of the Homography matrix to plot the super imposed image to the world frame.



## Getting Started

## Dependencies

* numpy library, python version 3.X.X needed before installing program.
* cv2 library, opencv is needed before running the program

### Installing

* Download the zip file \ pull all files and extract into any workspace as the code is self-contained and does not require any particular environment. 

### Executing program

* Open your IDE in the parent folder of detection.py run to visulizaing both April Tag detection and Superimposing of the testudo image onto the world frame.
* Open your text editor and ensure the python interpreter is at least python version 3.X.X and run the below comand (for VSC)
```
python3 detection.py
```

* Or open your text editor and ensure the python interpreter is at least python version 3.X.X and run the below comand (for VSC) 
```
CTRL + ALT + N
```

* Open your IDE in the parent folder of cube_projection.py and run to visulizaing Superimposing of the cube onto the world frame.
* Open your text editor and ensure the python interpreter is at least python version 3.X.X and run the below comand (for VSC) 
```
python3 cube_projection.py
```

* Or open your text editor and ensure the python interpreter is at least python version 3.X.X and run the below comand (for VSC)
```
CTRL + ALT + N
```
### Folder Structure:

```
📦AR-Tag-Detection-and-Augmented-Reality
 ┣ 📂Media
 ┃ ┣ 📜1tagvideo.mp4
 ┃ ┣ 📜kmatrix.xlsx - Sheet1.csv
 ┃ ┗ 📜testudo.png
 ┣ 📂Results
 ┃ ┣ 📂Detection _Testudo
 ┃ ┃ ┣ 📜Shi_tomasi_after_pt_selection.png
 ┃ ┃ ┣ 📜Shi_tomasi_pre_selection.png
 ┃ ┃ ┣ 📜camera_to_world.png
 ┃ ┃ ┣ 📜circular_inner_and_outter_mask.png
 ┃ ┃ ┣ 📜circular_outter_mask_only.png
 ┃ ┃ ┣ 📜encoder_img.png
 ┃ ┃ ┣ 📜fft_img.png
 ┃ ┃ ┣ 📜rectangular_mask.png
 ┃ ┃ ┣ 📜start_image.png
 ┃ ┃ ┗ 📜warp_world_to_camera.png
 ┃ ┣ 📜Cube_output.avi
 ┃ ┣ 📜cube_world_frame.png
 ┃ ┗ 📜testudo_output.avi
 ┣ 📂__pycache__
 ┃ ┣ 📜encoder.cpython-36.pyc
 ┃ ┗ 📜imagePreprocessing.cpython-36.pyc
 ┣ 📜README.md
 ┣ 📜cube_projection.py
 ┣ 📜detection.py
 ┣ 📜encoder.py
 ┗ 📜imagePreprocessing.py
```
## Authors

Sri Sai Charan Velisetti - svellise@umd.edu

