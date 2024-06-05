import numpy as np
import cv2
import easyocr
import os

reader = easyocr.Reader(['en'])

def load_images_from_folder(folder,dest,amount):
    image_paths = []
    count = 0
    for filename in os.listdir(folder):
        save = dest + filename
        read = folder + filename
        image_paths.append(filename)
        cv2.imwrite(save,clean_image(read))
        count = count + 1
        if count == amount:
            break
    return image_paths

def clean_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    scale_percent = 200
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
        if (x2 - x1) * (y2 - y1) < pixel_count/64:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,255), -1)

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
            if w*h < pixel_count/4:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255,255,255), -1)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_BINARY)[1]

    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    return image

load_images_from_folder('testdata/BarH/','graytest/BarH/',25)
load_images_from_folder('testdata/BarV/','graytest/BarV/',25)
load_images_from_folder('testdata/Line/','graytest/Line/',25)
load_images_from_folder('testdata/Pie/','graytest/Pie/',25)

