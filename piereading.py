import numpy as np
import cv2
import easyocr
import math

pixel_count = 0
reader = easyocr.Reader(['en'])
image = cv2.imread('data/Pie/1156.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

copy = cv2.resize(image, (600, 400))
cv2.imshow('Resized_Window', copy)
cv2.waitKey(0)

#Object detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

ythresh = 5
pairs = []
for box in box_list:
    boxx = int((2 * box[0] + box[2]) / 2)
    boxy = int((2 * box[1] + box[3]) / 2)
    cti = 0
    distance = 500000
    for textbox in result:
        textx = int((textbox[0][0][0] + textbox[0][2][0]) / 2)
        texty = int((textbox[0][0][1] + textbox[0][2][1]) / 2)
        dist = math.sqrt(abs(boxx - textx)**2 + abs(boxy - texty)**2)
        if (dist < distance and abs(boxy - texty) < ythresh and 
            abs(textbox[0][0][1] - textbox[0][2][1])* 2 > box[3]):
            distance = dist
            cti = textbox
    pairs.append([box,cti,0,0])

gray = cv2.cvtColor(copyimg, cv2.COLOR_BGR2GRAY)
for pair in pairs:
    hist,bins = np.histogram(gray[(pair[0][1]):(pair[0][1]+pair[0][3]), (pair[0][0]):(pair[0][0]+pair[0][2])].ravel(),256,[0,256])
    plist = []
    for a in range(len(hist)):
        plist.append([a,hist[a]])
    n = len(plist)
    for i in range(n):
        for j in range(n-1):
            if plist[j][1] > plist[j+1][1]:
                plist[j], plist[j+1] = plist[j+1], plist[j]
    if plist[255][0] != 255:
        pair[2] = plist[255]
    print(pair)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
copy = cv2.resize(gray, (600, 400))
cv2.imshow('Resized_Window', copy)
cv2.waitKey(0)

def condition(x):
    return x != 0
def condx(x):
    return x[2] != 0
count = sum(condition(x[2]) for x in pairs)

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

for pair in pairs:
    if pair[2] != 0:
        dst = 256
        for x in slicepixels:
            if abs(x[0]-pair[2][0]) < dst:
                dst = abs(x[0]-pair[2][0])
                pair[3] = x
                chosen = x
        slicepixels.remove(chosen) 

pairs = list(filter(condx, pairs)) 

sum = 0
for pair in pairs:
    sum = sum + pair[3][1]
for pair in pairs:
    print(pair[1][1]," = ",(pair[3][1] * 360 / piecount)) 
