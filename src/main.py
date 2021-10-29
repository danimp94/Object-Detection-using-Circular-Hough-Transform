import operator
import cv2 as cv
import numpy as np
import sys
import time
import numpy as np
import  itertools

def output_circles(image,x0,y0,r,color,thickness):
    for t in range(0,360):
        x=int(x0+r*(np.cos(np.radians(t))))
        y=int(y0+r*(np.sin(np.radians(t))))
        for d in range(thickness):
            image.itemset((y,x,0),color[0])
            image.itemset((y,x,1),color[1])
            image.itemset((y,x,2),color[2])
    cv.imshow("Circled Image",image)

def houghTra(sepX,sepY,sepR):
    accumulator = {}
    # Looping over the values of theta, Considering only even values for for minimizing performance time by setting increment as 2
    for t in range(0,360,2):
        # Determining all the possible centres x0,y0 using the above formula
        x0 = int(sepX-(sepR*np.cos(np.radians(t))))
        y0 = int(sepY-(sepR*np.sin(np.radians(t))))
        # Checking if the center is within the range of image
        if x0>0 and x0<width and y0>0 and y0<height:
            # Voting process...
            if (x0,y0,sepR) in accumulator:
                accumulator[(x0,y0,sepR)]=accumulator[(x0,y0,sepR)]+1
            else:
                accumulator[(x0,y0,sepR)]=0
    entire_sorted_accumulator = sorted(accumulator.items(),key=operator.itemgetter(1),reverse=True)
    return entire_sorted_accumulator[:5]

def pixiExt(img,x0,y0,radius):
    pixsLoca = []
    pixsTotal = 0
    totalInten = 0
    unionPix = 0
    unionInte = 0
    uPixVal = []
    
    height0,width0 = img.shape
    y, x = np.ogrid[:height0, :width0]
    
    dist_from_center = np.sqrt((x0-x)**2 + (y0-y)**2)
    mask0 = dist_from_center <= radius
    mask1 = dist_from_center <= radius*1.1
    mask2 = mask1 & ~mask0
    
    height1,width1 = mask0.shape
    for y, x in itertools.product(range(0,height1), range(0,width1)):
        if mask0[y,x].any() == True and x>=0 and x<=width and y>=0 and y<=height:
            totalInten += img[y,x].item()
            pixsTotal += 1
            pixsLoca.append((img[y,x].item()))
        if mask2[y,x].any() == True and x>=0 and x<=width and y>=0 and y<=height:
                unionPix += 1
                unionInte += img[y,x].item()
                uPixVal.append((img[y,x].item()))
    return pixsTotal,totalInten,pixsLoca,unionPix,unionInte,uPixVal

def paraCal(pR1,pR2,inR1,inR2,inValR1,inValR2,uPi,uIn,uPiVal):
    Areg = 0
    uIn = uIn   #union intensity
    uPi = uPi  #pixels in union
    
    if uIn and uPi != 0:
        uAvgIn = uIn/uPi #avage intensity of union
        for i in range(len(uPiVal)):
            Areg += ((uPiVal[i]-uAvgIn))**2 #The avage intensity of region R1
            #print(len(inValR1))
            
        Breg = (pR1*(((inR1/pR1)-uAvgIn)**2))+(pR2*(((inR2/pR2)-uAvgIn)**2))
        if Areg and Breg !=0:
            regRatio = Breg/Areg
            
            return regRatio    

if __name__ == "__main__":
    
    #img1
    filename = "coconut_3.jpg"
    img1 = cv.imread(filename)
    gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    dst = cv.Canny(gray,80,200)
    
    height,width = dst.shape
    Rmin = 10
    Rmax = 40
    highChanObj = []
    
    #print(pixiExt(dst,250,250,50))
    #time.sleep(10)
    #color: (0,255,0)
    
    for y, x, r in itertools.product(range(0,height,20), range(0,width,20),range(Rmin,Rmax,4)):
        if x-Rmin>=0 and x+Rmin<=width and y-Rmin>=0 and y+Rmin<=height:
            pixisR1,intenR1,pixInValR1,uPi,uIn,uInVal= pixiExt(dst,x,y,r)
            pixisR2,intenR2,pixInValR2,uPi,uIn,uInVal  = pixiExt(dst,x,y,r*1.1)
            regionRatio = paraCal(pixisR1,pixisR2,intenR1,intenR2,pixInValR1,pixInValR2,uPi,uIn,uInVal)
            
            height0,width0 = dst.shape
            y3, x3 = np.ogrid[:height0, :width0]
            
            dist_from_center = np.sqrt((x - x3)**2 + (y-y3)**2)
            mask0 = dist_from_center <= r
            mask1 = dist_from_center <= r*1.1
            mask2 = mask1 & ~mask0
            coppyimg = dst.copy()
            coppyimg[mask2] = 255
            
            cv.imshow("elCircle", coppyimg)
            cv.waitKey(1)
            
            if regionRatio != None and regionRatio > 0.75 and regionRatio < 2:
                highChanObj.append({x,y,r,regionRatio})
                print("\r",regionRatio)
                print(houghTra(x,y,r))
            
        sys.stdout.write("\r" + str(int(float(y+1)/float(height)*100))+"%")
        sys.stdout.flush()
    
    
