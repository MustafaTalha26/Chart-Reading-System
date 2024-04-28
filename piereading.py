import numpy as np
import cv2
import easyocr
import math

def checkfloat(string):
    try:
        string = float(string)
        return True
    except ValueError:
        return False
    
def correctgroups(result):
    thresh = 5
    correctresults = []
    for x in result:
        if x[2] > 0.30:
            correctresults.append(x)
    result = correctresults

    groups = []
    group = []        
    for chosen in result:
        group = []
        group.append(chosen)
        for other in result:
            if chosen != other and abs(chosen[0][0][0]-other[0][0][0]) < thresh:
                group.append(other)
        group.sort()
        if group not in groups and len(group) > 1:
            groups.append(group)
        group = []

    for chosen in result:
        group = []
        group.append(chosen)
        for other in result:
            if chosen != other and abs(chosen[0][0][1]-other[0][0][1]) < thresh:            
                group.append(other)
        group.sort()
        if group not in groups and len(group) > 1:
            groups.append(group)
        group = []

    for chosen in result:
        group = []
        group.append(chosen)
        for other in result:
            if chosen != other and abs(chosen[0][2][0]-other[0][2][0]) < thresh:
                group.append(other)
        group.sort()
        if group not in groups and len(group) > 1:
            groups.append(group)
        group = []

    for chosen in result:
        group = []
        group.append(chosen)
        for other in result:
            if chosen != other and abs(chosen[0][2][1]-other[0][2][1]) < thresh:
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

def read_pie_chart(imagepath,language='en'):
    pixel_count = 0
    reader = easyocr.Reader([language])
    image = cv2.imread(imagepath)

    #Scale the image
    scale_percent = 100 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    pixel_count = len(image) * len(image[0])
    copyimg = image.copy()

    # Read and remove text
    result = reader.readtext(image)
    for text in result:
        x1 = int(text[0][0][0])
        x2 = int(text[0][2][0])
        y1 = int(text[0][0][1]) 
        y2 = int(text[0][2][1])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,255), -1)

    ngroups,lgroups = correctgroups(result)
    longlg = lgroups[0]
    for group in lgroups:
        if len(longlg) < len(group):
            longlg = group  
    colorboxes = []
    seperateimg = image.copy()
    for word in longlg:
        squ = abs(word[0][3][1]-word[0][0][1])
        colorbox = image[word[0][0][1]:word[0][2][1],word[0][0][0]-int((15*squ)/10):word[0][2][0]-int((15*squ)/10)].copy()
        cv2.rectangle(seperateimg, (word[0][0][0]-int((15*squ)/10), word[0][0][1]), (word[0][2][0]-int((15*squ)/10), word[0][2][1]), (255,255,255), -1)
        erosion_kernel = np.ones((5, 15), np.uint8) 
        colorbox = cv2.erode(colorbox, erosion_kernel)
        colorbox = cv2.cvtColor(colorbox, cv2.COLOR_RGB2GRAY)
        sumgray = 0
        sumn = 0
        for x in range(len(colorbox)):
            for y in range(len(colorbox[0])):
                if (colorbox[x][y] != 255 ):
                    sumgray = sumgray + colorbox[x][y]
                    sumn = sumn + 1
        if sumn == 0:
            print("A word invaded legendbox names. Please lower the threshold of OCR grouping function")
            exit()
        sumgray= int(sumgray / sumn)
        colorboxes.append([word,sumgray,0])

    copy = cv2.resize(image, (600, 400))
    cv2.imshow('Resized_Window', copy)
    cv2.waitKey(0)

    #Object detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_BINARY)[1]

    erosion_kernel = np.ones((3, 3), np.uint8) 
    eroded = cv2.erode(thresh, erosion_kernel)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,3))
    close = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    dilate = cv2.dilate(close, dilate_kernel, iterations=1)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    box_list = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 150 and area < 15000:
            x,y,w,h = cv2.boundingRect(c)
            if w*h < pixel_count/4:
                box_list.append([x,y,w,h])
                cv2.rectangle(image, (x - 2, y - 2), (x + w + 2, y + h + 2), (255,255,255), -1)

    copy = cv2.resize(image, (600, 400))
    cv2.imshow('Resized_Window', copy)
    cv2.waitKey(0)

    def condition(x):
        return x != 0
    def condx(x):
        return x[1] != 0
    count = sum(condition(x[1]) for x in colorboxes)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist,bins = np.histogram(gray.ravel(),256,[0,256])
    plist = []
    for a in range(len(hist)):
        plist.append([a,hist[a]])
    n = len(plist)
    for i in range(n):
        for j in range(n-1):
            if plist[j][1] > plist[j+1][1]:
                plist[j], plist[j+1] = plist[j+1], plist[j]

    slicepixels = []
    if plist[255][0] == 255:
        for x in range(count):
            slicepixels.append(plist[254-x])
    piecount = sum(x[1] for x in slicepixels)

    for pair in colorboxes:
        if pair[1] != 0:
            dst = 256
            for x in slicepixels:
                if abs(x[0]-pair[1]) < dst:
                    dst = abs(x[0]-pair[1])
                    pair[2] = x
                    chosen = x
            slicepixels.remove(chosen) 

    colorboxes = list(filter(condx, colorboxes)) 

    sumnum = 0
    for pair in colorboxes:
        sumnum = sumnum + pair[2][1]
    for pair in colorboxes:
        print(pair[0][1]," = ",(pair[2][1] * 360 / piecount)) 
