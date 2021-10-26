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
height,width = img_2.shape

radii = 50
radius = 30

acc_array = np.zeros(((height,width,radii)))

filter3D = np.zeros((30,30,radii))
filter3D[:,:,:]=1

x0 = 20
y0 = 20

x = radius
y=0
decision = 1-x

edges = np.where(img_2==255)

for i in range(0,len(edges[0])):
    x=edges[0][i]
    y=edges[1][i]
    for radius in range(20,55):
        

        while(y<x):  
            if(x + x0<height and y + y0<width):
                acc_array[ x + x0,y + y0,radius]+=1; # Octant 1
            if(y + x0<height and x + y0<width):
                acc_array[ y + x0,x + y0,radius]+=1; # Octant 2
            if(-x + x0<height and y + y0<width):
                acc_array[-x + x0,y + y0,radius]+=1; # Octant 4
            if(-y + x0<height and x + y0<width):
                acc_array[-y + x0,x + y0,radius]+=1; # Octant 3
            if(-x + x0<height and -y + y0<width):
                acc_array[-x + x0,-y + y0,radius]+=1; # Octant 5
            if(-y + x0<height and -x + y0<width):
                acc_array[-y + x0,-x + y0,radius]+=1; # Octant 6
            if(x + x0<height and -y + y0<width):
                acc_array[ x + x0,-y + y0,radius]+=1; # Octant 8
            if(y + x0<height and -x + y0<width):
                acc_array[ y + x0,-x + y0,radius]+=1; # Octant 7
            y+=1
            if(decision<=0):
                decision += 2 * y + 1
            else:
                x=x-1;
                decision += 2 * (y - x) + 1

i=0
j=0            

while(i<height-30):
    while(j<width-30):
        filter3D=acc_array[i:i+30,j:j+30,:]*filter3D
        max_pt = np.where(filter3D==filter3D.max())
        a = max_pt[0].astype(int)       
        b = max_pt[1].astype(int)
        c = max_pt[2].astype(int)

        b=b+j
        a=a+i
        if(filter3D.max()>90):
            #cv.circle(output,(b,a),c,(0,255,0),cv.FILLED)
            cv.circle(output,(b,a),5,(0,255,0),2)
        j=j+30
        filter3D[:,:,:]=1
    j=0
    i=i+30

cv.imshow('Detected circle',output)

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
