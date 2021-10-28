import cv2 as cv
import numpy as np
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

def pixiExt(img,centx0,centy0,radius):
    pixsTotal = 0
    totalInten = 0
    
    #file1 = open("pixels.txt","w")
    height0,width0 = img.shape
    y, x = np.ogrid[:height0, :width0]
    
    dist_from_center = np.sqrt((x - centx0)**2 + (y-centy0)**2)
    mask = dist_from_center <= radius
    #file1.write(str(intenReg))
    #file1.close()
    
    height1,width1 = mask.shape
    for i in range(height1):
        for j in range(width1):
            if mask[i,j].any() == True:
                totalInten += img[i,j].item()
                pixsTotal += 1
    return pixsTotal,totalInten

if __name__ == "__main__":
    
    #img1
    filename = "aau-city-1.jpg"
    img1 = cv.imread(filename)
    gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    dst = cv.Canny(gray,80,200)
        
    print(pixiExt(dst,100,100,50))
