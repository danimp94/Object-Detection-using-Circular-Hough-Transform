import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Input Image
filename = ("data\coconut_1.jpg")
img = cv.imread(filename)

# Enhancing Process

    # Portable Gray Map
img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Enhancing Image:
    # Histogram Equalization 
img_hist = cv.equalizeHist(img_grey)
    # Canny edge detection
img_edge = cv.Canny(img_hist,200,600) #Default parameters:(100,200)
    # Closing and Opening Morphology
img_dil = cv.dilate(
    img_edge,
    cv.getStructuringElement(cv.MORPH_RECT, (2, 2)),
    iterations=1
)

img_ero = cv.erode(
    img_dil,
    cv.getStructuringElement(cv.MORPH_RECT, (2, 2)),
    iterations=1
)


## Separability Filter
#def separability_filter():
    
## CHT
#def circle_hough_transform():
    

#input_image(filename)
#preprocessing_image(img,0)

cv.imshow('img',img)
cv.imshow('img_grey',img_grey)
cv.imshow('img_histogram_equalization',img_hist)
cv.imshow('img_canny_edge',img_edge)
cv.imshow('img_dilate',img_dil)
cv.imshow('img_erode',img_ero)
cv.waitKey(0)
