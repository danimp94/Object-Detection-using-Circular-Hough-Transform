import cv2 as cv
import numpy as np
import sys
import time
import numpy as np
import  itertools
from numpy.core.records import array
np.set_printoptions(threshold=sys.maxsize)

def pixiExt(img,centx0,centy0,radius):
    pixsLoca = []
    coppyimg = img.copy()
    pixsTotal = 0
    totalInten = 0
    
    #file1 = open("pixels.txt","w")
    height0,width0 = img.shape
    y, x = np.ogrid[:height0, :width0]
    
    dist_from_center = np.sqrt((x - centx0)**2 + (y-centy0)**2)
    mask = dist_from_center <= radius
    #file1.write(str(intenReg))
    #file1.close()
    #coppyimg[mask] = [0]
    #cv.imshow("elCircle", coppyimg)
    #cv.waitKey(0)
    
    height1,width1 = mask.shape
    for y, x in itertools.product(range(0,height1), range(0,width1)):
        if mask[y,x].any() == True and x>0 and x<width and y>0 and y<height:
            totalInten += img[y,x].item()
            pixsTotal += 1
            pixsLoca.append((img[y,x].item()))
            
    return pixsTotal,totalInten,pixsLoca

def paraCal(pR1,pR2,inR1,inR2,inValR1,inValR2):
    AregR1 = 0
    AregR2 = 0
    Areg = 0
    uIn = inR2-inR1 #union intensity
    uPi = pR2-pR1   #pixels in union
    
    if uIn and uPi != 0:
        uAvgIn = uIn/uPi #avage intensity of union
        for i in range(len(inValR1)):
            AregR1 += ((inValR1[i]-uAvgIn))**2 #The avage intensity of region R1
            #print(len(inValR1))
            
        for i in range(len(inValR2)):
            AregR2 += ((inValR2[i]-uAvgIn))**2 #The avage intensity of region R2
            #print(len(inValR2))
            
        Areg = AregR1 + AregR2
        Breg = (pR1*(((inR1/pR1)-uAvgIn)**2))+(pR2*(((inR2/pR2)-uAvgIn)**2))
        if Areg and Breg !=0:
            regRatio = Breg/Areg
            
            return regRatio    

if __name__ == "__main__":
    
    #img1
    filename = "pipes.jpg"
    img1 = cv.imread(filename)
    gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    dst = cv.Canny(gray,80,200)
    
    height,width = dst.shape
    coppyimg = dst.copy()
    Rmin = 10
    Rmax = 100
    regionRatioContainer = []
    
    #print(pixiExt(dst,250,250,50))
    #time.sleep(10)
    
    for y, x, r in itertools.product(range(0,height,50), range(0,width,50),range(Rmin,Rmax,20)):
        if dst.item(y,x) == 255 and x-Rmax>0 and x+Rmax<width and y-Rmax>0 and y+Rmax<height:
            pixisR1,intenR1,pixInValR1 = pixiExt(dst,x,y,r)
            pixisR2,intenR2,pixInValR2  = pixiExt(dst,x,y,r*1.25)
            regionRatio = paraCal(pixisR1,pixisR2,intenR1,intenR2,pixInValR1,pixInValR2)
            
            height0,width0 = dst.shape
            y3, x3 = np.ogrid[:height0, :width0]
            
            dist_from_center = np.sqrt((x3 - x)**2 + (y3-y)**2)
            mask0 = dist_from_center <= r
            coppyimg[mask0] = 255
            cv.imshow("elCircle", coppyimg)
            cv.waitKey(0)
            regionRatioContainer.append(regionRatio)
            print("\r",regionRatio)
            
        sys.stdout.write("\r" + str(int(float(y+1)/float(height)*100))+"%")
        sys.stdout.flush()
