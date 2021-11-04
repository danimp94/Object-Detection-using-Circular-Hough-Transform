import operator
import cv2 as cv
import numpy as np
import sys
import numpy as np
import  itertools

def drawCicle(img,finale):
    x = 0
    y = 0
    x = 0
    img1 = img.copy()
    for i in range(len(finale)):
        x = finale[i][0]
        y = finale[i][1]
        r = finale[i][2]
        img1 = cv.circle(img,(x,y),r,255,5)
        
    #status = cv.imwrite('/home/rz/Documents/uniwork/Vision/MiniPro/Res/img.png',img1)
    #print(status)
    return cv.imshow("Circles found",img1)

def houghTra(img,sepX,sepY,sepR):
    accumulator = {}
    imgnewp = img.copy()
    # Looping over the values of theta, Considering only even values for for minimizing performance time by setting increment as 2
    for t in range(0,360,2):
        # Determining all the possible centres x0,y0 using the above formula
        x0 = int(sepX+(sepR*np.cos(np.radians(t))))
        y0 = int(sepY+(sepR*np.sin(np.radians(t))))
        # Checking if the center is within the range of image
        
        """ imgnewnp = cv.circle(imgnewp,(x0,y0),sepR,255,5)
        cv.imshow("hough circles", imgnewnp)
        cv.waitKey(1) """
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
    circleOnTop = True
    ij = 0
    ji = 0
    circleInRang = True
    ii = 0
    jj = 0
    finCic = []
    jCicle = []
    solodolo = []
    imgnew = dst.copy()
    iii = 0

    #print("OG list",sortArr)
    
    """ while circleOnTop:
        #print("ji and ij",ji,ij,len(sortArr))
        if sortArr[ij][0][0][0] - sortArr[ji][0][0][0] == 0 and ij != ji:
            #print("it do go down")
            if sortArr[ij][0][0][2] > sortArr[ji][0][0][2]:
                #print("this will removed ij>",sortArr[ji])
                sortArr.remove(sortArr[ji])
                if ij >= int(len(sortArr)):
                    circleOnTop = False
                    break
                
            if sortArr[ji][0][0][2] > sortArr[ij][0][0][2]:
                #print("this will removed ji>",sortArr[ij])
                sortArr.remove(sortArr[ij])
                if ij >= int(len(sortArr)):
                    circleOnTop = False
                    break
                
            if sortArr[ij][0][0][2] == sortArr[ji][0][0][2]:
                #print("this gets removed ==",sortArr[ji])
                sortArr.remove(sortArr[ji])
                if ij >= int(len(sortArr)):
                    circleOnTop = False
                    break
        ji += 1
        #print("ji and ij",ji,ij,len(sortArr))
        if ji >= int(len(sortArr)):
            ij += 1
            ji = 0
            
        if ij >= int(len(sortArr)):
            circleOnTop = False
            break """
            
    #print("\r on top circles gone", sortArr)
    """ while iii < len(sortArr): #for worksheet
        imgnewn = cv.circle(imgnew,(sortArr[iii][0][0][0],sortArr[iii][0][0][1]),sortArr[iii][0][0][2],255,5)
        cv.imshow("all circles", imgnewn)
        cv.waitKey(1)
        iii +=1 """
        
    while circleInRang:
        #print("ii,jj,len()",ii,jj,len(sortArr))
        #print("ii,jj",ii,jj,len(sortArr))
        dist = np.linalg.norm(np.array((int(sortArr[ii][0][0][0]),int(sortArr[ii][0][0][1])))-np.array((int(sortArr[jj][0][0][0]),int(sortArr[jj][0][0][1]))))
        #if ((sortArr[ii][0][0][0] - sortArr[jj][0][0][0] <= 10) or (sortArr[ii][0][0][0] - sortArr[jj][0][0][0] >= -10)) and ii != jj:
        if dist <= Rmax*1.7 and ii != jj:
            jCicle.append(sortArr[jj])
            #print("jcicle",jCicle,ii,jj,dist)
            sortArr.remove(sortArr[jj])
            jj = jj-2
            
            if ii >= int(len(sortArr)):
                #print("are you here dwag?")
                circleInRang = False
                break
            
        jj += 1
        #print("jj",jj)
        if jj >= int(len(sortArr)):
            avgX = 0
            avgY = 0
            maxR = 0
            rdiMax = 0
            
            if len(jCicle) != 0:
                for i in range(len(jCicle)):
                    #print("range af circle",len(jCicle))
                    avgX += int(jCicle[i][0][0][0])
                    avgY += int(jCicle[i][0][0][1])
                    maxR = int(jCicle[i][0][0][2])
                    
                    if maxR >= rdiMax:
                        rdiMax = maxR
                    
                #print("all of", avgX,avgY,maxR,jCicle)
                avgX = int(avgX/len(jCicle))
                avgY = int(avgY/len(jCicle))
                #maxR = int(maxR/len(jCicle))
                finCic.append([avgX,avgY,rdiMax])
                #print("avg loop fincic",finCic)
                jCicle = []
            else:
                avgX = int(sortArr[ii][0][0][0])
                avgY = int(sortArr[ii][0][0][1])
                maxR = int(sortArr[ii][0][0][2])
                solodolo.append([avgX,avgY,maxR])
                #print("add if not within dist",finCic,dist)
            ii += 1
            jj = 0
            #print("in reset ii and jj",ii,jj,len(sortArr))
            if ii >= int(len(sortArr)):
                #print("are you here dwag?")
                circleInRang = False
                break
            
        
    #print("it do be done",sortArr)
    #print("fincic",finCic)
    return finCic

if __name__ == "__main__":
    
    ## Enhancing Process
    # PREPROCESSING:
    #img1 
    filename = "test3.jpg"
    img1 = cv.imread(filename)
    imgcopy = img1.copy()

    # Portable Gray Map
    img_grey = cv.cvtColor(imgcopy, cv.COLOR_BGR2GRAY)

    # Enhancing Image:
    img_hist = cv.equalizeHist(img_grey) # Histogram Equalization 

    # Canny edge detection
    img_edge = cv.Canny(img_hist,400,50) #Default parameters:(100,200) #test1 and 2 400,50,1-1,1-1

    # Closing and Opening Morphology
    img_dil = cv.dilate(
        img_edge,
        cv.getStructuringElement(cv.MORPH_RECT, (1, 1)),
        iterations=1)
    img_ero = cv.erode(
        img_dil,
        cv.getStructuringElement(cv.MORPH_RECT, (1, 1)),
        iterations=1)
    dst = img_ero
    
    cv.imshow("is it okay?", dst)
    cv.waitKey(0)
    
    height,width = dst.shape
    Rmin = 53#11 test2 #17 test1
    Rmax = 118#47 #38
    highChanObj = []
    finale0 = []
    itDoBeDone = []
    
    for y, x, r in itertools.product(range(0,height,20), range(0,width,20),range(Rmin,Rmax,5)):
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
                cv.imshow("Circle with separability over the threshold", imgnewn)
                cv.waitKey(1)
            
        sys.stdout.write("\r" + str(int(float(y+1)/float(height)*100))+"% picture scaned")
        sys.stdout.flush()
    finale0 = circles(highChanObj,Rmax)
    for i in range(len(finale0)):
        pixisR1,intenR1,pixInValR1,uPi,uIn  = pixiExt(dst,finale0[i][0],finale0[i][1],finale0[i][2])
        pixisR2,intenR2,pixInValR2,uPi,uIn  = pixiExt(dst,finale0[i][0],finale0[i][1],finale0[i][2]*1.25)
        regionRatio = paraCal(pixisR1,pixisR2,intenR1,intenR2,pixInValR1,pixInValR2,uPi,uIn)
        print(regionRatio)
        if regionRatio != None and regionRatio > 0.10:
            itDoBeDone.append([finale0[i][0],finale0[i][1],finale0[i][2]])
    drawCicle(dst,itDoBeDone)
    drawCicle(imgcopy,itDoBeDone)
    cv.waitKey(0)
