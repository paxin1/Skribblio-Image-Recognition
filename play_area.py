import numpy as np
import cv2 as cv
import pyautogui
import keyboard
from matplotlib import pyplot as plt
from matplotlib import colors

def take_screenshot():
    image = pyautogui.screenshot()
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    return image

def find_draw_area(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    blur = cv.medianBlur(img,9)

    light_background_hsv = (100,180,135)
    dark_background_hsv = (120,230,155)
    hsv_blur =  cv.cvtColor(blur, cv.COLOR_RGB2HSV)

    background_mask = cv.inRange(hsv_blur, light_background_hsv, dark_background_hsv)
    inverted_mask = cv.bitwise_not(background_mask)

    result = cv.bitwise_and(blur, blur, mask=background_mask)

    img_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    img_gray_blur = cv.medianBlur(img_gray, 15)
    thresh = cv.adaptiveThreshold(img_gray_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    img_h, img_w, _ = img.shape
    upper_area_bound = img_h * img_w / 2
    draw_area = contours[0]
    max_area = 0

    for contour in contours:
        area = cv.contourArea(contour)
        if area > 20000 and area > max_area and area < upper_area_bound:
            draw_area = contour
            max_area = area

    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    rect = cv.boundingRect(draw_area)
    x,y,w,h = rect
    box = cv.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
    cropped = img[y: y+h, x: x+w]
    cv.imwrite("drawing.png", cropped)

def find_current_play_area():
    screenshot = take_screenshot()
    find_draw_area(screenshot)