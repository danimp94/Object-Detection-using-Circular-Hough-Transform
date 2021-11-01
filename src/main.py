import operator
import cv2 as cv
import numpy as np
import sys
import time
import numpy as np
import  itertools

def drawCicle(img,finale):
    x = 0
    y = 0
    x = 0

    for i in range(len(finale)):
        x = finale[i][0]
        y = finale[i][1]
        r = finale[i][2]

        img1 = cv.circle(img,(x,y),r,255,5)

    return cv.imshow("Circles found",img1)
 
    
def output_circles(image,x0,y0,r,color,thickness):
    for t in range(0,360):
        x=int(x0+r*(np.cos(np.radians(t))))
        y=int(y0+r*(np.sin(np.radians(t))))
        for d in range(thickness):
            image.itemset((y,x,0),color[0])
            image.itemset((y,x,1),color[1])
            image.itemset((y,x,2),color[2])
    return cv.imshow("Circled Image",image)

def houghTra(img,sepX,sepY,sepR):

    accumulator = {}
    imgnewp = img.copy()
    # Looping over the values of theta, Considering only even values for for minimizing performance time by setting increment as 2
    for t in range(0,360,2):
        # Determining all the possible centres x0,y0 using the above formula
        x0 = int(sepX-(sepR*np.cos(np.radians(t))))
        y0 = int(sepY-(sepR*np.sin(np.radians(t))))
        # Checking if the center is within the range of image
        
        #imgnewnp = cv.circle(imgnewp,(x0,y0),sepR,255,5)
        #cv.imshow("hough circles", imgnewnp)
        #cv.waitKey(1)
        if x0>0 and x0<width and y0>0 and y0<height:
            # Voting process...
            if (x0,y0,sepR) in accumulator:
                accumulator[(x0,y0,sepR)]=accumulator[(x0,y0,sepR)]+1
            else:
                accumulator[(x0,y0,sepR)]=0
    entire_sorted_accumulator = sorted(accumulator.items(),key=operator.itemgetter(1),reverse=True)

    return  entire_sorted_accumulator[0]

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
    mask1 = dist_from_center <= radius*1.25
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
    return pixsTotal,totalInten,pixsLoca,unionPix,unionInte

def paraCal(pR1,pR2,inR1,inR2,inValR1,inValR2,uPi,uIn):

    Areg = 0
    uIn = uIn   #union intensity
    uPi = uPi  #pixels in union
    
    if uIn and uPi != 0:
        uAvgIn = uIn/uPi #avage intensity of union
        for i in range(len(inValR1)):
            Areg += ((inValR1[i]-uAvgIn))**2 #The avage intensity of region R1
            #print(len(inValR1))
            
        for i in range(len(inValR2)):
            Areg += ((inValR2[i]-uAvgIn))**2 #The avage intensity of region R1
            #print(len(inValR1))
            
        Breg = (pR2*(((inR2/pR2)-uAvgIn)**2))+(pR1*(((inR1/pR1)-uAvgIn)**2))
        if Areg and Breg !=0:
            regRatio = Breg/Areg
            
            return regRatio    

def circles(array,Rmax):

    sortArr = sorted(array, key=operator.itemgetter(0,0,0,0))     
    preX = 0
    preY = 0
    currX = 0
    currY = 0
    r1 = 0
    r2 = 0
    finaleCircle = []
    goody = False
    distGood = False
    addonces = False
    #storeCircler = []
    onlyOnce = True
    storeO = False

    for i in range(len(sortArr)):
        onlyOnce = True
        for j in range(len(sortArr)):
            preX = int(sortArr[i][0][0][0])
            preY = int(sortArr[i][0][0][1])
            currX = int(sortArr[j][0][0][0])
            currY = int(sortArr[j][0][0][1])
            r1 = int(sortArr[i][0][0][2])
            r2 = int(sortArr[j][0][0][2])

            if i == j:
                break
            print("dis dist sucks dist", np.linalg.norm(np.array((preX,preY))-np.array((currX,currY))))
            if np.linalg.norm(np.array((preX,preY))-np.array((currX,currY))) >= 40:
                goody = True
                distGood = True
            else:
                distGood = False
            if r1 == r2 and onlyOnce and np.linalg.norm(np.array((preX,preY))-np.array((currX,currY))) >= 40:
                print("store circlereaekfpoaef")
                #storeCircler.append([sortArr[i][0][0][0],sortArr[i][0][0][1],sortArr[i][0][0][2]])
                onlyOnce = False
                goody = True
                storeO = True
                break
            if r1 == r2:
                goody = False
                break
            if r1 < r2 and not distGood:
                print("tessttss")
                goody = False                    
                break
            else:
                print("goody is good")
                print("r1 og r2", r1,r2)
                goody = True
        if goody:
            print("pending your mom")
            finaleCircle.append([sortArr[i][0][0][0],sortArr[i][0][0][1],sortArr[i][0][0][2]])

    print(finaleCircle)
    return finaleCircle

if __name__ == "__main__":
    
    #img1
    filename = "pipes3.jpg"
    img1 = cv.imread(filename)
    imgCopy3 = img1.copy()
    gray = cv.cvtColor(imgCopy3,cv.COLOR_BGR2GRAY)
    dst = cv.Canny(imgCopy3,80,300)
    
    height,width = dst.shape
    Rmin = 11
    Rmax = 37
    highChanObj = []
    finale0 = []
    
    for y, x, r in itertools.product(range(0,height,10), range(0,width,10),range(Rmin,Rmax,5)):
        if x-Rmin>=0 and x+Rmin<=width and y-Rmin>=0 and y+Rmin<=height:
            pixisR1,intenR1,pixInValR1,uPi,uIn  = pixiExt(dst,x,y,r)
            pixisR2,intenR2,pixInValR2,uPi,uIn  = pixiExt(dst,x,y,r*1.25)
            regionRatio = paraCal(pixisR1,pixisR2,intenR1,intenR2,pixInValR1,pixInValR2,uPi,uIn)
            #print(r)
            height0,width0 = dst.shape
            y3, x3 = np.ogrid[:height0, :width0]
            
            dist_from_center = np.sqrt((x - x3)**2 + (y-y3)**2)
            mask0 = dist_from_center <= r
            mask1 = dist_from_center <= r*1.25
            mask2 = mask1 & ~mask0
            coppyimg = dst.copy()
            coppyimg[mask2] = 255
            
            cv.imshow("elCircle", coppyimg)
            cv.waitKey(1)
            
            if regionRatio != None and regionRatio > 0.10:

                houghTra(dst,x,y,r)
                highChanObj.append([houghTra(dst,x,y,r)])
                #print("\r B/A",regionRatio)
              
                imgnew = dst.copy()
                imgnewn = cv.circle(imgnew,(x,y),r,255,5)
                cv.imshow("optimal circle", imgnewn)
                cv.waitKey(1)
            
        sys.stdout.write("\r" + str(int(float(y+1)/float(height)*100))+"%")
        sys.stdout.flush()

    finale0 = circles(highChanObj,Rmax)
    drawCicle(dst,finale0)
    cv.waitKey(0)
