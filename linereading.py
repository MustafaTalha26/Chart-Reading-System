import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import easyocr
from skimage.transform import (hough_line, hough_line_peaks)
import Kmeans
import csv

# Necessary Functions
def checkfloat(string):
    try:
        string = float(string)
        return True
    except ValueError:
        return False

def sort_by_indexes(lst, indexes, reverse=False):
    return [val for (_, val) in sorted(zip(indexes, lst), key=lambda x: \
            x[0], reverse=reverse)]

def medyanVer(nlist):
    mid = len(nlist) // 2
    if len(nlist) % 2 == 1:
        res = (float(nlist[mid][1]) - float(nlist[mid-1][1])) / (nlist[mid][0][0][1] - nlist[mid-1][0][0][1])
    if len(nlist) % 2 == 0:
        res = (float(nlist[mid][1]) - float(nlist[~mid][1])) / (nlist[mid][0][0][1] - nlist[~mid][0][0][1])
    return res
def medyanHor(nlist):
    mid = len(nlist) // 2
    if len(nlist) % 2 == 1:
        res = (float(nlist[mid][1]) - float(nlist[mid-1][1])) / (nlist[mid][0][0][0] - nlist[mid-1][0][0][0])
    if len(nlist) % 2 == 0:
        res = (float(nlist[mid][1]) - float(nlist[~mid][1])) / (nlist[mid][0][0][0] - nlist[~mid][0][0][0])
    return res

def maxVer(nlist,topleft,y_scale):
    mid = len(nlist) // 2
    if len(nlist) % 2 == 1:
        res = float(nlist[mid-1][1]) + y_scale * (topleft - nlist[mid][0][0][1])
    if len(nlist) % 2 == 0:
        res = float(nlist[~mid][1]) + y_scale * (topleft - nlist[~mid][0][0][1])
    return res
def maxHor(nlist,bottomright,x_scale):
    mid = len(nlist) // 2
    if len(nlist) % 2 == 1:
        res = float(nlist[mid-1][1]) + x_scale * (bottomright - nlist[mid-1][0][0][0])
    if len(nlist) % 2 == 0:
        res = float(nlist[~mid][1]) + x_scale * (bottomright - nlist[~mid][0][0][0])
    return res

# Grouping up number groups and letter groups by their vertical or horizontal neighborhood
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

# Function for creating csv file
def read_line_graph(image_path,language=['en'],frequency=30,kmeans_colors_extra = 3):
    reader = easyocr.Reader(language)
    image = cv2.imread(image_path)
    
    # Amount of data points going to be extracted is not certain
    # Can be increased to acquire more data points
    frequency = frequency

    # There can be noise on image of the chart
    # This value can be changed if many noise are expected
    kmeans_colors_extra = kmeans_colors_extra

    # Image scaling will be needed for EasyOCR to read better
    scale_percent = 200 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # Threshold for minimum chart area
    # Chart can not be smaller than quarter of the image
    pixel_count = len(image) * len(image[0])
    contourthresh = pixel_count/4

    # Saving up EasyOCR output and removing text
    result = reader.readtext(image)
    for text in result:
        x1 = int(text[0][0][0])
        x2 = int(text[0][2][0])
        y1 = int(text[0][0][1]) 
        y2 = int(text[0][2][1])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,255), -1)

    # Number groups and letter groups identifed but needs further selection
    ngroups,lgroups = correctgroups(result)

    # We need X and Y axis numbers to read chart
    # Number groups can't be less
    if len(ngroups) < 2:
        print("Number groups not sufficent")
        exit()

    # Number group selection for longest groups
    # This will help us recover X and Y axis groups
    firstng = ngroups[0]
    secondng = ngroups[1]
    for group in ngroups:
        if len(group) > len(firstng):
            secondng = firstng
            firstng = group
        elif len(group) < len(firstng) and len(group) > len(secondng):
            secondng = group

    # For identifying chart boundries
    # Only bottomright and topleft numbers taken
    # Other numbers will be provided from hough line transformation
    bottomrightcorner = 0
    topleftcorner = len(image)
    mixed = firstng + secondng
    for x in mixed:
        if x[0][1][0] > bottomrightcorner:
            bottomrightcorner = x[0][1][0]
        if x[0][1][1] < topleftcorner:
            topleftcorner = x[0][1][1]

    # Scale variables will be usefull for finding point values
    # Image points and real chart values will be identified with these numbers.
    x_scale = 0
    x_max = 0
    y_scale = 0
    y_max = 0

    # These section will choose which numbergroup belongs to X and which to Y axis
    # Scale variables will be assigned here
    if abs(firstng[0][0][0][0] - firstng[1][0][0][0]) < 7 or abs(firstng[0][0][2][0] - firstng[1][0][2][0]) < 7:
        if abs(secondng[0][0][0][1] - secondng[1][0][0][1]) < 7 or abs(secondng[0][0][2][1] - secondng[1][0][2][1]) < 7:
            firstsorted = []
            for x in firstng:
                firstsorted.append(float(x[0][0][1]))
            firstsorted = sort_by_indexes(firstng,firstsorted)
            secondsorted = []
            for x in secondng:
                secondsorted.append(float(x[0][0][0]))
            secondsorted = sort_by_indexes(secondng,secondsorted)
            x_scale = medyanHor(secondsorted)
            y_scale = medyanVer(firstsorted)
            x_max = maxHor(secondsorted,bottomrightcorner,x_scale)
            y_max = maxVer(firstsorted,topleftcorner,y_scale)
            print(x_scale, y_scale, x_max, y_max)
        else:
            print("Both number groups are vertical.")
            exit()
    if abs(firstng[0][0][0][1] - firstng[1][0][0][1]) < 7 or abs(firstng[0][0][2][1] - firstng[1][0][2][1]) < 7:
        if abs(secondng[0][0][0][0] - secondng[1][0][0][0]) < 7 or abs(secondng[0][0][2][0] - secondng[1][0][2][0]) < 7:
            firstsorted = []
            for x in firstng:
                firstsorted.append(float(x[0][0][0]))
            firstsorted = sort_by_indexes(firstng,firstsorted)
            secondsorted = []
            for x in secondng:
                secondsorted.append(float(x[0][0][1]))
            secondsorted = sort_by_indexes(secondng,secondsorted)
            x_scale = medyanHor(firstsorted)
            y_scale = medyanVer(secondsorted)
            x_max = maxHor(firstsorted,bottomrightcorner,x_scale)
            y_max = maxVer(secondsorted,topleftcorner,y_scale)
            print(x_scale, y_scale, x_max, y_max)
        else:
            print("Both number groups are horizontal.")
            exit()

    # Letter group is explanation of which color belong to which line
    # We choose longest letter group for further use
    longlg = lgroups[0]
    for group in lgroups:
        if len(longlg) < len(group):
            longlg = group  

    # Colors are nearby letters
    # We identify these colors with mean values of colorbox
    # Word and its color are ready for further use
    colorboxes = []
    for word in longlg:
        squ = abs(word[0][3][1]-word[0][0][1])
        colorbox = image[word[0][0][1]:word[0][2][1],word[0][0][0]-int((15*squ)/10):word[0][2][0]-int((15*squ)/10)].copy()
        cv2.rectangle(image, (word[0][0][0]-int((15*squ)/10), word[0][0][1]), (word[0][2][0]-int((15*squ)/10), word[0][2][1]), (255,255,255), -1)
        erosion_kernel = np.ones((5, 15), np.uint8) 
        colorbox = cv2.erode(colorbox, erosion_kernel)
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
        if sumn == 0:
            print("A word invaded legendbox names. Please lower the threshold of OCR grouping function")
            exit()
        sumcolorred = int(sumcolorred / sumn)
        sumcolorgreen= int(sumcolorgreen / sumn)
        sumcolorblue= int(sumcolorblue / sumn)
        colorbox = [sumcolorred,sumcolorgreen,sumcolorblue]
        colorboxes.append([word,colorbox])

    # This section is for clearing leftovers from chart
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

    for c in cnts:
        area = cv2.contourArea(c)
        if area > 200 and area < 15000:
            x,y,w,h = cv2.boundingRect(c)
            if w*h < contourthresh:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255,255,255), -1)

    copy = cv2.resize(image, (600, 400))
    cv2.imshow('Resized_Window', copy)
    cv2.waitKey(0)

    # This section used for preparing image to find X and Y axis using Hough Line 
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_BINARY)[1]

    erosion_kernel = np.ones((3, 3), np.uint8) 
    eroded = cv2.erode(thresh, erosion_kernel)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    close = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    hough = cv2.dilate(close, dilate_kernel, iterations=1)

    copy = cv2.resize(hough, (600, 400))
    cv2.imshow('Resized_Window', copy)
    cv2.waitKey(0)

    # This section will create extra lines to easly identify X and Y axis
    # Helper of actual Hough Line section
    lines = cv2.HoughLines(hough, 1, np.pi / 180, 500, None)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(hough, pt1, pt2, (255,255,255), 3, cv2.LINE_AA)

    copy = cv2.resize(hough, (600, 400))
    cv2.imshow('Resized_Window', copy)
    cv2.waitKey(0)   

    # This section used for finding X and Y axis using Hough Line
    # Line data will be extracted and chart's important image coordinates will be acquired
    hspace, theta, dist = hough_line(hough)
    h, q, d = hough_line_peaks(hspace, theta, dist)
    linesdata = zip(h,q,d)

    lineanglethresh = 0.1
    xaxis = [1.570,0]
    yaxis = [0.008,len(image)]
    for _,y,z in linesdata:
        if abs(abs(1.570) - abs(y)) < lineanglethresh and abs(z) > abs(xaxis[1]):
            xaxis[0] = y
            xaxis[1] = z
        if abs(abs(0.008) - abs(y)) < lineanglethresh and z < yaxis[1]:
            yaxis[0] = y
            yaxis[1] = z

    # This section will cut of chart from rest of the image
    if xaxis[1] != 0 and yaxis[1] != len(image):
        cutimg = image[topleftcorner:int((19*abs(xaxis[1]))/20), int((21*yaxis[1])/20):bottomrightcorner]
        erosion_kernel = np.ones((5, 15), np.uint8) 
        cutimg = cv2.erode(cutimg, erosion_kernel)
        copy = cv2.resize(cutimg, (600, 400))
        cv2.imshow('Resized_Window', copy)
        cv2.waitKey(0)
    else:
        print("X and Y axis can not be found")
        exit()

    # Only used for debugging
    # Can be cleared later
    angle_list=[]  
    fig, axes = plt.subplots(1, 3)
    ax = axes.ravel()

    ax[0].imshow(hough , cmap='gray')
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(np.log(1 + hspace),
                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), dist[-1], dist[0]],
                 cmap='gray', aspect=1/1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')
    ax[2].imshow(hough, cmap='gray')
    origin = np.array((0, hough.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(hspace, theta, dist)):
        angle_list.append(angle) 
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle+0.008)
        ax[2].plot(origin, (y0, y1), '-r')
    ax[2].set_xlim(origin)
    ax[2].set_ylim((hough.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    origin = np.array((0, hough.shape[1]))
    dist = xaxis[1]
    angle = xaxis[0]
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle+0.008)
    ax[2].plot(origin, (y0, y1), '-b')

    dist = yaxis[1]
    angle = yaxis[0]
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle+0.008)
    ax[2].plot(origin, (y0, y1), '-b')

    plt.tight_layout()
    plt.show()

    # This section used for creating artifical pillars to extract data points
    # Intersection of pillars and lines are data points that will be extracted
    # Clump is a data point with many color pixels
    # Clumps are data points in one pillar
    # Accur array of pillars 
    accur = []
    clump = []
    clumps = []
    clumpflag = 0
    for column in range(frequency):
        pillar = int(len(cutimg[0])/frequency) * column
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

    # After points extracted new data structure created as listed:
    # Y cord of image, X cord of image, RGB colors, group number
    # Group number is -1 by default
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
            groupsum = [int(ycords/lenclump),pixel[2],[int(sumclump[0]/lenclump),int(sumclump[1]/lenclump),int(sumclump[2]/lenclump)],-1]
            column.append(groupsum)
        columns.append(column)

    # RGB colors of letter's colorbox
    data = []
    for x in colorboxes:
        data.append([x[1][0],x[1][1],x[1][2]])

    # RGB colors of extracted data points
    points = []
    for column in columns:
        for point in column:
            points.append(point)
            data.append(point[2])
            cutimg[point[0],point[1]] = [0,0,0]

    # Letter group length + expected noise will determine how many color groups are on chart
    # Kmeans alghoritm used for grouping line color with data point colors
    # Alghoritm helps with fluctuating color values into reasonable groups
    kgroups = len(longlg) + kmeans_colors_extra
    colorNameAndGroup = []
    predictions = Kmeans.kmeansT(data,kgroups)
    for x in range(len(colorboxes)):
        colorNameAndGroup.append([colorboxes[x][0][1],predictions[x],[]])
    for x in range(len(points)):
        points[x][3] = predictions[x+len(colorboxes)]

    # Data points image cordinates transforms into real chart values
    for column in columns:
        for point in column:
            for x in colorNameAndGroup:
                if point[3] == x[1]:
                    x[2].append([(point[1]*x_scale),(y_max+(point[0]*y_scale))])

    for x in colorNameAndGroup:
        print(x[0])
        for point in x[2]:
            print(point)        

    copy = cv2.resize(cutimg, (600, 400))
    cv2.imshow('Resized_Window', copy)
    cv2.waitKey(0)

    with open('output.csv', 'w') as file:
        csv_writer = csv.writer(file)
        for x in colorNameAndGroup:
            csv_writer.writerow(x[0])
            for point in x[2]:
                csv_writer.writerow(point)  
