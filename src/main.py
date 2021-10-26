import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Input Image
filename = ("data\coconut_2.jpg")
img = cv.imread(filename)
output = img.copy()

# PREPROCESSING:

    # Portable Gray Map
img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Enhancing Image:
    # Histogram Equalization 
img_hist = cv.equalizeHist(img_grey)

    # Canny edge detection
img_edge = cv.Canny(img_hist,180,350) #Default parameters:(100,200)

    # Closing and Opening Morphology
img_dil = cv.dilate(
    img_edge,
    cv.getStructuringElement(cv.MORPH_RECT, (3, 3)),
    iterations=1
)
img_ero = cv.erode(
    img_dil,
    cv.getStructuringElement(cv.MORPH_RECT, (3, 3)),
    iterations=1
)
img_dil2 = cv.dilate(
    img_ero,
    cv.getStructuringElement(cv.MORPH_RECT, (2, 2)),
    iterations=1
)
img_2 = img_dil


# SEPARABILITY FILTER:
def sep_filter(img, r, r1, r2):

    # r: radius of center circle 
    # r1: inner radius (wi)
    # r2: outer radius (wo)

    # Obtain picture shape
    height,width = img_2.shape
    print(height,width)

    X = double(img)
    MAP = np.zeros(size(X)) ##????

    ## Define radius
    #r2 = 10 # Set of radius for R2:{r2,r2+10,...,ru}
    #r1 = r2*0.8 # R2 is R1 + 1/4*R1

    L = 2 * (r + r2) +1
    c = r + r2 + 1
    
    N1 = 0
    N2 = 0
    List1 = np.zeros(L^2,2) ## check
    List2 = np.zeros(L^2,2) ## check
    mask = np.zeros(L,L)

    for i in range(1,L):
        for j in range(1,L):
            if((r^2) >= ((c-i)^2 + (c-j)^2)) and ((r-r1)^2 <= ((c-i)^2 + (c-j)^2)):
                mask[i,j] = 0.5
                N1 = N1 + 1
                List1[N1,:] = [i,j]
            elif((r+r2)^2 >= ((c-j)^2 + (c-i)^2)) and ((r^2) <= ((c-j)^2 + (c-i)^2)):
                mask[i,j] = 1
                N2 = N2 + 1
                List2[N2,:] = [i,j]
    i=0
    j=0  

    List1 = List1[1:N1,:]
    List2 = List2[1:N2,:]
    N = N1 + N2
    V1 = np.zeros(N1,1)
    V2 = np.zeros(N2,1)

    for i in range(1,height-L+1):
       for j in range(1,widht-L+1):
          for l in range(1,size(List1[1])): ##CHECK
             V1[l] = X(List1(l, 1) + i - 1, List1(l, 2) + j - 1)
          for l in range(1,size(List2[1])): ##CHECK
             V2[l] = X(List2(l, 1) + i - 1, List2(l, 2) + j - 1)
          M = np.mean([V1,V2])
          St = np.cov([V1,V2],1)
          if (St == 0):
             MAP[i + c - 1, j + c - 1] = 0
          else:
             M1 = np.mean(V1)
             M2 = np.mean(V2)
             Sb = ((N1*((M1-M)^2)) + (N2*((M2-M)^2)))/N
             MAP[i + c - 1, j + c - 1] = Sb/St


sep_filter(img,9,8,10)
print(MAP)


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
cv.imshow('img_dilate_2',img_dil2)
cv.waitKey(0)
