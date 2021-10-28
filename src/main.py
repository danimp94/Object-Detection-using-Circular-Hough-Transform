import cv2 as cv
import numpy as np
import sys
import time
import numpy as np
import  itertools
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
    
    height,width = dst.shape
    x,y = np.ogrid[:height,:width]
    Rmin = 30
    Rmax = 60
    AFromPaper = 0
    WeirdNcontainer = []
    
    for y in range(0,height,20):
        for x in range(0,width,20):
            if dst.item(y,x) == 255:
                for r in range(Rmin,Rmax,1):
                    
                    pixisR1,intenR1 = pixiExt(dst,x,y,r)
                    pixisR2,intenR2 = pixiExt(dst,x,y,Rmax)
                    print("p1",pixisR1)
                    print("p2",pixisR2)
                    print("i1",intenR1)
                    print("i2",intenR2)
                    
                    unionInten = intenR2-intenR1
                    unionPixis = pixisR2-pixisR1
                    unionAvgInten = unionInten/unionPixis
                    print("union inten",unionInten)
                    print("union pix",unionPixis)
                    print("union avg int",unionAvgInten)
                    #time.sleep(5)
                    
                    AFromPaper = (unionInten-unionAvgInten)**2
                    BFromPaper = pixisR1*((intenR1/pixisR1)-unionAvgInten)+pixisR2*((intenR2/pixisR2)-unionAvgInten)
                    WeirdN = BFromPaper/AFromPaper
                    WeirdNcontainer.append(WeirdN)
                    
                    print("A", AFromPaper)
                    print("B", BFromPaper)
                    print("n",WeirdN)
