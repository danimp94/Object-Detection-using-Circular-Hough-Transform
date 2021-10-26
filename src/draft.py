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


##########################################3
#######################################
############################################
###########################################3

from __future__ import division
import cv2
import numpy as np
import time

original_image = cv2.imread('aau-city-1.jpg')
#gray_image = cv2.imread("aau-city-1.jpg",0)
cv2.imshow('Original Image',original_image)

output = original_image.copy()

#Gaussian Blurring of Gray Image
blur_image = cv2.GaussianBlur(original_image,(3,3),1)
cv2.imshow('Gaussian Blurred Image',blur_image)

#Using OpenCV Canny Edge detector to detect edges
edged_image = cv2.Canny(blur_image,100,100)
cv2.imshow('Edged Image', edged_image)

height,width = edged_image.shape
radii = 100

acc_array = np.zeros(((height,width,radii)))

filter3D = np.zeros((30,30,radii))
filter3D[:,:,:]=1

start_time = time.time()

def fill_acc_array(x0,y0,radius):
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
                

cv2.imshow('Detected circle',output)
end_time = time.time()
time_taken = end_time - start_time
print('Time taken for execution',time_taken)

cv2.waitKey(0)
cv2.destroyAllWindows()


function MAP = cvtCircleSepFilter(img, r, wi, wo)
#% Function to generate separability map using a circular mask filter
#% Input: img: gray-scale image 
#%        r: radius of center circle 
#%        wi: inner radius (described as r1 in the reference paper [1])
#%        wo: outer radius (described as r2 in the reference paper [1])
#% Output: MAP: separability map
#%
#% If you use this code, we would appreciate if you cite the following paper:

[H, W] = size(img);###
X = double(img);
MAP = zeros(size(X));
L = 2 * (r + wo) +1; ###
c = r+wo+1; ###
N1 = 0;
N2 = 0;
List1=zeros(L^2,2);
List2=zeros(L^2,2);
MASK = zeros(L,L);
for py=1:L;
   for px = 1:L;
      if (r^2) >= ((c-py)^2 + (c-px)^2) && (r-wi)^2 <= ((c-py)^2 + (c-px)^2)
         MASK(py,px) = 0.5;
         N1 = N1 + 1;
         List1(N1,:) =[py,px];
      elseif (r+wo)^2 >= ((c-py)^2 + (c-px)^2) && (r^2) <= ((c-py)^2 + (c-px)^2)
         MASK(py,px) = 1;
         N2 = N2 + 1;
         List2(N2,:) =[py,px];
      end
   end
end


List1 = List1(1:N1,:);
List2 = List2(1:N2,:);
N = N1 + N2;
V1 = zeros(N1,1);
V2 = zeros(N2,1);
for y=1:H-L+1;
   for x=1:W-L+1;
      for l = 1:size(List1,1)
         V1(l) = X(List1(l,1)+y-1,List1(l,2)+x-1);
      end
      for l = 1:size(List2,1)
         V2(l) = X(List2(l,1)+y-1,List2(l,2)+x-1);
      end
      M = mean([V1;V2]);
      St = cov([V1;V2],1);
      if St == 0
         MAP(y+c-1,x+c-1) = 0;
      else
         M1 = mean(V1);
         M2 = mean(V2);
         Sb = ((N1*((M1-M)^2)) + (N2*((M2-M)^2)))/N;
         MAP(y+c-1,x+c-1) = Sb/St;
      end
   end
end
