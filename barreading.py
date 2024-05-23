from collections import defaultdict
import cv2
import numpy as np
import easyocr
import statistics

def findNumberValueOCR(image, bar):
    if bar == 1:
        image = cv2.resize(image, (800, 1250))
    else:
        image = cv2.resize(image, (2200, 800))

    str_gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = str_gray_image.shape[:2]
    if bar == 1:
        clippedImage = str_gray_image[int(height // 1.2):, :]
    else:
        clippedImage = str_gray_image[:, :int(height // 2.8)]

    thresh = cv2.adaptiveThreshold(clippedImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 89, 68)

    height_1, width_1 = thresh.shape[:2]

    reader = easyocr.Reader(['en'])
    results = reader.readtext(thresh)

    max_count = defaultdict(int)
    for bbox, text, prob in results:
        if bar == 1:
            location = bbox[1][1]
            max_count[location] += 1
        else:
            location = bbox[1][0]
            max_count[location] += 1

    bars_location = max(max_count, key=max_count.get)

    number_list = {}
    for (bbox, text, prob) in results:
        if (bar == 1):
            if bbox[1][1] >= (bars_location - 3) and bbox[1][1] <= (bars_location + 3):
                number_list[text] = bbox[1][0]
        else:
            if bbox[1][0] >= (bars_location - 3) and bbox[1][0] <= (bars_location + 3):
                number_list[text] = bbox[1][1]

    valid_number_list = {}
    min_subs = 0
    temp = 0
    for i, k in number_list.items():
        try:
            number = int(i)
            valid_number_list[number] = k
            if (temp == 0):
                temp = number
                min_subs = temp
            else:
                if ((number - temp) < min_subs):
                    min_subs = number - temp
                temp = number
        except ValueError:
            continue

    bar_values = []
    temp_number = 0
    temp_bar = 0
    for i, y in valid_number_list.items():
        if (temp_number == 0):
            temp_number = i
            temp_bar = y
        else:
            bar_values.append((y - temp_bar) / ((i - temp_number) / min_subs))

    ortalama = statistics.mean(bar_values)

    if (bar == 1):
        ortalama = ortalama / width_1
    else:
        ortalama = ortalama / height_1

    return ortalama,abs(min_subs)
def find_barNames_OCR(image, bar):
    reader = easyocr.Reader(['en'])

    if(bar == 1):
        rotation_matrix = image
    else:
        rotation_matrix = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    str_gray_image = cv2.cvtColor(rotation_matrix, cv2.COLOR_BGR2GRAY)
    height, width = str_gray_image.shape[:2]

    if(bar == 1):
        clippedImage = str_gray_image[:, :int(width // 2.6)]
    else:
        clippedImage = str_gray_image[:, int(width // 1.6):]

    height_1, width_1= clippedImage.shape[:2]

    results = reader.readtext(clippedImage, detail=3)

    max_count = defaultdict(int)

    for bbox, text, prob in results:
        if bar == 1:
            location = bbox[1][0]
            max_count[location] += 1
        else:
            location = bbox[0][0]
            max_count[location] += 1

    bars_location = max(max_count, key=max_count.get)

    word_list = {}
    for (bbox, text, prob) in results:
        if(bar == 1):
            if bbox[1][0] >= (bars_location-3) and bbox[1][0] <= (bars_location+3):
                word_list[text] = bbox[1][1]/height_1
        else:
            if bbox[0][0] >= (bars_location-3) and bbox[0][0] <= (bars_location+3):
                word_list[text] = 1 - (bbox[0][1]/height_1)

    return word_list
def mappingNamesValues(bar_data, word_list):
    new_values = {}
    bur_number_border = { 1: 0.105, 2: 0.095, 3: 0.085, 4:0.075, 5:0.065, 6:0.055, 7:0.045 , 8:0.035, 9: 0.025, 10:0.015}

    if len(word_list) <= 7:
        border = bur_number_border[len(word_list)]
    else:
        border = 0.035

    for name_key, name_value in word_list.items():
        for bar_key, bar_value in bar_data.items():
            if(abs(bar_key - name_value) < border):
                new_values[name_key] = round(bar_value)
            else:
                new_values.setdefault(name_key, 0)

    return new_values

def horizontal_bar_graph(imagePath):
    image = cv2.imread(imagePath)
    new_width = 800
    new_height = 600
    resized_image = cv2.resize(image, (new_width, new_height))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(thresh, kernel)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    close = cv2.morphologyEx(eroded_image, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    dilated_image = cv2.dilate(close, kernel, iterations=8)

    mask = cv2.inRange(dilated_image, 0, 200)

    height, width = mask.shape[:2]
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    temp_data = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        temp_bottom_left = (x, y + h)
        temp_data.append(temp_bottom_left[0])
    most_common_element = max(set(temp_data), key=temp_data.count)

    value_tuple = findNumberValueOCR(image,1)
    bar_data = {}
    for i in range(len(contours)):
        m = max(contours[i][0])
        x, y, w, h = cv2.boundingRect(contours[i])
        bottom_left = (x, y + h)
        if (bottom_left[0] > most_common_element - 5 and bottom_left[0] < most_common_element + 5):
            value = ((w/width)/value_tuple[0])*value_tuple[1]
            if (int(value) != 0):
                bar_data[m[1] / height] = int(value) + (value_tuple[1] * 0.15)

    bar_data = dict(reversed(bar_data.items()))
    bar_names = find_barNames_OCR(image, 1)
    bar_names = dict(sorted(bar_names.items(), key=lambda x: x[1]))

    new_bars = mappingNamesValues(bar_data,bar_names)

    for x,y in new_bars.items():
        print(x,": ",y)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def vertical_bar_graph(imagePath):
    image = cv2.imread(imagePath)
    new_width = 800
    new_height = 600
    resized_image = cv2.resize(image, (new_width, new_height))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(thresh, kernel)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    close = cv2.morphologyEx(eroded_image, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    dilated_image = cv2.dilate(close, kernel, iterations=8)
    mask = cv2.inRange(dilated_image, 0, 200)

    height, width = mask.shape[:2]
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    temp_data = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        temp_bottom_left = (x, y + h)
        temp_data.append(temp_bottom_left[1])
    most_common_element = max(set(temp_data), key=temp_data.count)

    value_tuple = findNumberValueOCR(image,2)
    bar_data = {}
    for i in range(len(contours)):
        m = max(contours[i][0])
        x, y, w, h = cv2.boundingRect(contours[i])
        bottom_left = (x, y + h)
        if(bottom_left[1] > most_common_element-5 and bottom_left[1] < most_common_element+5):
            value = ((h / height) / value_tuple[0]) * value_tuple[1]
            bar_data[m[0] / width] = int(value) + (value_tuple[1] * 0.15)

    bar_names = find_barNames_OCR(image,2);
    bar_names = dict(sorted(bar_names.items(), key=lambda x: x[1]))
    bar_data = dict(sorted(bar_data.items(), key=lambda x: x[0]))
    new_bars = mappingNamesValues(bar_data,bar_names)

    for x,y in new_bars.items():
        print(x,": ",int(y))

    cv2.waitKey(0)
    cv2.destroyAllWindows()





