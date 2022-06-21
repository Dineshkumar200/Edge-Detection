# Edge-Detection
## Aim:
To perform edge detection using Sobel, Laplacian, and Canny edge detectors.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import the required packages for further process.


### Step2:
Read the image and convert the bgr image to gray scale image.

### Step3:
Use any filters for smoothing the image to reduse the noise.

### Step4:
Apply the respective filters -Sobel,Laplacian edge dectector and Canny edge dector.

### Step5:
Display the filtered image using plot and imshow.

 
## Program:

``` Python
# Import the packages

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image, Convert to grayscale and remove noise


ip_img=cv2.imread("parrot.jpg")
gray_img=cv2.cvtColor(ip_img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Canny edge detector",gray_img)
cv2.waitKey(0)
img=cv2.GaussianBlur(gray_img,(3,3),0)


# SOBEL EDGE DETECTOR

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
sobelxy = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)

plt.figure(1)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.show()

plt.figure(1)
plt.subplot(2,2,1),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobelx'), plt.xticks([]), plt.yticks([])
plt.show()

plt.figure(1)
plt.subplot(2,2,1),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobely'), plt.xticks([]), plt.yticks([])
plt.show()

plt.figure(1)
plt.subplot(2,2,1),plt.imshow(sobelxy,cmap = 'gray')
plt.title('Sobelxy'), plt.xticks([]), plt.yticks([])
plt.show()

# LAPLACIAN EDGE DETECTOR


laplacian = cv2.Laplacian(img,cv2.CV_64F)

cv2.imshow("Laplacian edge detector",laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()


# CANNY EDGE DETECTOR

canny = cv2.Canny(img1, 70, 150)
cv2.imshow("Canny edge detector",canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Output:
### SOBEL EDGE DETECTOR
![di1](https://user-images.githubusercontent.com/75235789/168871322-8af3ee77-0813-422e-bd0a-c12cd9409436.jpg)


### LAPLACIAN EDGE DETECTOR
![di2](https://user-images.githubusercontent.com/75235789/168871365-f902b403-3c06-4064-a7b8-610b8ebb7b9b.jpg)


### CANNY EDGE DETECTOR
![di3](https://user-images.githubusercontent.com/75235789/168871380-8b6db311-c468-4d00-8c30-f1b63ef5fdeb.jpg)


## Result:
Thus the edges are detected using Sobel, Laplacian, and Canny edge detectors.
