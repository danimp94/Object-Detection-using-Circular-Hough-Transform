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

def hog_trans(edged_image, x0, y0,radius):
    
    height,width = edged_image.shape
    radii = 100

    acc_array = np.zeros(((height,width,radii)))

    filter3D = np.zeros((30,30,radii))
    filter3D[:,:,:]=1

    start_time = time.time()

    x = radius
    y=0
    decision = 1-x
    
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
    
    
    edges = np.where(edged_image==255)
    for i in range(0,len(edges[0])):
        x=edges[0][i]
        y=edges[1][i]
        for radius in range(20,55):
            fill_acc_array(x,y,radius)
            

    i=0
    j=0
    while(i<height-30):
        while(j<width-30):
            filter3D=acc_array[i:i+30,j:j+30,:]*filter3D
            max_pt = np.where(filter3D==filter3D.max())
            a = max_pt[0]       
            b = max_pt[1]
            c = max_pt[2]
            b=b+j
            a=a+i
            if(filter3D.max()>90):
                cv2.circle(output,(b,a),c,(0,255,0),2)
            j=j+30
            filter3D[:,:,:]=1
        j=0
        i=i+30
    
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