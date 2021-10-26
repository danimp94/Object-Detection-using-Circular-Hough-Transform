import numpy as np


# Input Image
# Enhancing Process
# Portable Gray Map
# Enhancing Image:
# Histogram Equalization 
# Canny edge detection
# Closing and Opening Morphology

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def hog_trans(img_egded, rho=180, theta=180):
    
    
    return 

if __name__ == "__main__":

    img1 = cv2.imread("aau-city-1.jpg")
    img_grey1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img_hist1 = cv2.equalizeHist(img_grey1)
    img_gau1 = cv2.GaussianBlur(img_hist1,(3,3),1)
    img_edge1 = cv2.Canny(img_gau1,100,200)

    img_di1 = cv2.dilate(
        img_edge1,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )

    img_ero1 = cv2.erode(
        img_di1,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )


    cv2.imshow("img_grey1",img_ero1)
    cv2.waitKey(0)
