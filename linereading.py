import numpy as np
import cv2
import matplotlib.pyplot as plt
import easyocr
from collections import Counter 
from skimage.transform import (hough_line, hough_line_peaks)
import DBSCAN

pixel_count = 0
reader = easyocr.Reader(['en'])
image = cv2.imread('data/Line/99968.png')

def checkfloat(string):
    try:
        string = float(string)
        return True
    except ValueError:
        return False

def medyan(nlist):
    nlist.sort()
    mid = len(nlist) // 2
    res = (nlist[mid] + nlist[~mid]) / 2
    return res

def mode(nlist):
    data = Counter(nlist) 
    get_mode = dict(data) 
    mode = [k for k, v in get_mode.items() if v == max(list(data.values()))]
    return mode 

def correctgroups(result):
    correctresults = []
    for x in result:
        if x[2] > 0.40:
            correctresults.append(x)
    result = correctresults

    groups = []
    group = []        
    for chosen in result:
        group = []
        group.append(chosen)
        for other in result:
            if chosen != other and abs(chosen[0][0][0]-other[0][0][0]) < 10:
                group.append(other)
        group.sort()
        if group not in groups and len(group) > 1:
            groups.append(group)
        group = []

    for chosen in result:
        group = []
        group.append(chosen)
        for other in result:
            if chosen != other and abs(chosen[0][0][1]-other[0][0][1]) < 10:            
                group.append(other)
        group.sort()
        if group not in groups and len(group) > 1:
            groups.append(group)
        group = []

    for chosen in result:
        group = []
        group.append(chosen)
        for other in result:
            if chosen != other and abs(chosen[0][2][0]-other[0][2][0]) < 10:
                group.append(other)
        group.sort()
        if group not in groups and len(group) > 1:
            groups.append(group)
        group = []

    for chosen in result:
        group = []
        group.append(chosen)
        for other in result:
            if chosen != other and abs(chosen[0][2][1]-other[0][2][1]) < 10:
                group.append(other)
        group.sort()
        if group not in groups and len(group) > 1:
            groups.append(group)
        group = []

    finegroups = []
    flag = 0
    for group in groups:
        flag = 0
        for other in groups:
            if other != group and all(x in other for x in group):
                flag = 1
        if flag == 0:
            finegroups.append(group)
    groups = finegroups

    numbergroups = []
    flag = 0
    for group in groups:
        flag = 0
        for x in group:
            if checkfloat(x[1]) == False:
                flag = 1
        if flag == 0:
            numbergroups.append(group)

    lettergroups = []
    flag = 0
    for group in groups:
        flag = 0
        for x in group:
            if checkfloat(x[1]) == True:
                flag = 1
        if flag == 0:
            lettergroups.append(group)
    
    return numbergroups,lettergroups

scale_percent = 200 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

pixel_count = len(image) * len(image[0])

result = reader.readtext(image)
for text in result:
    x1 = int(text[0][0][0])
    x2 = int(text[0][2][0])
    y1 = int(text[0][0][1]) 
    y2 = int(text[0][2][1])
    cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,255), -1)

#düzelt
ngroups,lgroups = correctgroups(result)
firstng = ngroups[0]
secondng = ngroups[1]
for group in ngroups:
    if len(group) > len(firstng):
        secondng = firstng
        firstng = group
    elif len(group) < len(firstng) and len(group) > len(secondng):
        secondng = group
    
if len(ngroups) == 2:
    xmid = []
    ymid = []
    for x in firstng:
        xmid.append(float(x[1]))
    for x in secondng:
        ymid.append(float(x[1]))
    xmid = medyan(xmid)
    ymid = medyan(ymid)

# düzelt
longlg = lgroups[0]
for group in lgroups:
    if len(longlg) < len(group):
        longlg = group     
colorboxes = []
for word in longlg:
    squ = abs(word[0][3][1]-word[0][0][1])
    colorbox = image[word[0][0][1]:word[0][2][1],word[0][0][0]-int((15*squ)/10):word[0][2][0]-int((15*squ)/10)]
    sumcolorred = 0
    sumcolorgreen = 0
    sumcolorblue = 0
    sumn = 0
    for x in range(len(colorbox)):
        for y in range(len(colorbox[0])):
            if True != (colorbox[x][y][0] == 255 and colorbox[x][y][1] == 255 and colorbox[x][y][2] == 255):
                sumcolorred = sumcolorred + colorbox[x][y][0]
                sumcolorgreen = sumcolorgreen + colorbox[x][y][1]
                sumcolorblue = sumcolorblue + colorbox[x][y][2]
                sumn = sumn + 1
    sumcolorred = int(sumcolorred / sumn)
    sumcolorgreen= int(sumcolorgreen / sumn)
    sumcolorblue= int(sumcolorblue / sumn)
    colorbox = [sumcolorred,sumcolorgreen,sumcolorblue]
    colorboxes.append([word,colorbox])

pre_gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_BINARY)[1]

erosion_kernel = np.ones((2, 2), np.uint8) 
eroded = cv2.erode(thresh, erosion_kernel)

close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
close = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, close_kernel, iterations=2)

dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
dilate = cv2.dilate(close, dilate_kernel, iterations=1)

cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

contourthresh = pixel_count/4
for c in cnts:
    area = cv2.contourArea(c)
    if area > 200 and area < 15000:
        x,y,w,h = cv2.boundingRect(c)

        if w*h < contourthresh:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255,255,255), -1)
        if w*h > contourthresh:
            print("Cords of big contour: ",x,y,x+w,y+h)

copy = cv2.resize(image, (600, 400))
cv2.imshow('Resized_Window', copy)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_BINARY)[1]

erosion_kernel = np.ones((2, 2), np.uint8) 
eroded = cv2.erode(thresh, erosion_kernel)

close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
close = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, close_kernel, iterations=1)

dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
hough = cv2.dilate(close, dilate_kernel, iterations=1)

tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
hspace, theta, dist = hough_line(hough , tested_angles)
h, q, d = hough_line_peaks(hspace, theta, dist)
linesdata = zip(h,q,d)

lineanglethresh = 0.2
xaxis = [0,0]
yaxis = [1.6, 100000]
for _,y,z in linesdata:
    if abs(0.008 - y) < lineanglethresh and z < yaxis[1]:
        yaxis[0] = y
        yaxis[1] = z
    elif abs(1.570 - y) < lineanglethresh and z > xaxis[1]:
        xaxis[0] = y
        xaxis[1] = z
print(xaxis)
print(yaxis)

if xaxis[1] != 0 and yaxis[1] != 100000:
    cutimg = image[0:int(xaxis[1])-3, int(yaxis[1])+3:len(image[1])]
    copy = cv2.resize(cutimg, (600, 400))
    cv2.imshow('Resized_Window', copy)
    cv2.waitKey(0)

#angle_list=[]  
#fig, axes = plt.subplots(1, 3)
#ax = axes.ravel()
#
#ax[0].imshow(hough , cmap='gray')
#ax[0].set_title('Input image')
#ax[0].set_axis_off()
#
#ax[1].imshow(np.log(1 + hspace),
#             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), dist[-1], dist[0]],
#             cmap='gray', aspect=1/1.5)
#ax[1].set_title('Hough transform')
#ax[1].set_xlabel('Angles (degrees)')
#ax[1].set_ylabel('Distance (pixels)')
#ax[1].axis('image')
#ax[2].imshow(hough, cmap='gray')
#origin = np.array((0, hough.shape[1]))
#for _, angle, dist in zip(*hough_line_peaks(hspace, theta, dist)):
#    angle_list.append(angle) 
#    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
#    ax[2].plot(origin, (y0, y1), '-r')
#ax[2].set_xlim(origin)
#ax[2].set_ylim((hough.shape[0], 0))
#ax[2].set_axis_off()
#ax[2].set_title('Detected lines')
#
#origin = np.array((0, hough.shape[1]))
#dist = xaxis[1]
#angle = xaxis[0]
#y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
#ax[2].plot(origin, (y0, y1), '-b')
#
#dist = yaxis[1]
#angle = yaxis[0]
#y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
#ax[2].plot(origin, (y0, y1), '-b')
#
#plt.tight_layout()
#plt.show()

accur = []
clump = []
clumps = []
clumpflag = 0
dots = 30
for column in range(dots):
    pillar = int(len(cutimg[0])/dots) * column
    for x in range(len(cutimg[:,pillar])):
        if True != (cutimg[:,pillar][x][0] == 255 and cutimg[:,pillar][x][1] == 255 and cutimg[:,pillar][x][2] == 255) :
            clump.append([x,[cutimg[:,pillar][x][0],cutimg[:,pillar][x][1],cutimg[:,pillar][x][2]],pillar])
            clumpflag = 1
        elif cutimg[:,pillar][x][0] == 255 and cutimg[:,pillar][x][1] == 255 and cutimg[:,pillar][x][2] == 255 and clumpflag == 1:
            clumps.append(clump)
            clumpflag = 0
            clump = []
    accur.append(clumps)
    clumps = []

print(len(cutimg[0]), len(cutimg))

columns = []
for clumps in accur:
    column = []
    for clump in clumps:
        sumclump = [0,0,0]
        lenclump = len(clump)
        ycords = 0
        for pixel in clump:
            sumclump[0] = sumclump[0] + pixel[1][0]
            sumclump[1] = sumclump[1] + pixel[1][1]
            sumclump[2] = sumclump[2] + pixel[1][2]
            ycords = ycords + pixel[0]
        RG = int(sumclump[0]/lenclump) - int(sumclump[1]/lenclump)
        GB = int(sumclump[1]/lenclump) - int(sumclump[2]/lenclump)
        RB = int(sumclump[0]/lenclump) - int(sumclump[2]/lenclump)
        groupsum = [int(ycords/lenclump),pixel[2],[int(sumclump[0]/lenclump),int(sumclump[1]/lenclump),int(sumclump[2]/lenclump)],[RG,GB,RB],-1]
        column.append(groupsum)
    columns.append(column)

points = []
data = []
for x in colorboxes:
    RG = x[1][0] - x[1][1]
    GB = x[1][1] - x[1][2]
    RB = x[1][0] - x[1][2]
    print([RG,GB,RB])
    data.append([RG,GB,RB])

for column in columns:
    for point in column:
        points.append(point)
        data.append(point[3])
        cutimg[point[0],point[1]] = [0,0,0]

predictions = DBSCAN.dbscanT(data,5)
for x in range(len(colorboxes)):
    print(predictions[x])
for x in range(len(points)):
    points[x][4] = predictions[x+len(colorboxes)]

for column in columns:
    print('New Column: ')
    for point in column:
        #if True != (point[3][0] == 0 and point[3][1] == 0 and point[3][2] == 0) and point[4] != -1:
        print('Data Points: ', point)

copy = cv2.resize(cutimg, (600, 400))
cv2.imshow('Resized_Window', copy)
cv2.waitKey(0)

