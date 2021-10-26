import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

filename = ("..\data\coconut_1.jpg")
img = cv.imread(filename2)

cv.imshow('img',img)

# Input Image
def input_image(filename):

    img = cv.imread(filename)

# Enhancing Process
def preprocessing_image(image):

    # Portable Gray Map
    img_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Enhancing Image:
    # Histogram Equalization 
    img_hist = cv.equalizeHist(img_grey)
    # Canny edge detection
    # Closing and Opening Morphology

    return img_hist

## Separability Filter
#def separability_filter():
    
## CHT
#def circle_hough_transform():
    

#input_image(filename)
preprocessing_image(img)

cv.imshow(img_hist)
cv.waitKey(0)
