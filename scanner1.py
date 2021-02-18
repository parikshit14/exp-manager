import cv2
from PIL import Image
import tempfile
import numpy as np


def process_image_for_ocr(file_path):
# TODO : Implement using opencv
    temp_filename = set_image_dpi(file_path)
    im_new = remove_noise_and_smooth(temp_filename)
    return im_new


def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = max(1, int(1080/ length_x))
    size = factor * length_x, factor * width_y
    # size = (1800, 1800)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    #cv2.imshow("dpi",im_resized)
    #cv2.waitKey(0)
    im_resized.show("dpi")
    return temp_filename


def image_smoothening(img):
    cv2.imshow("before smooth",img)
    cv2.waitKey(0)
    """ret1, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)"""
    height,width=img.shape
    gaussian_filter_img = cv2.GaussianBlur(img,(11,11),0)
    cv2.imshow("smooth",gaussian_filter_img)
    cv2.waitKey(0)
    return gaussian_filter_img


def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("closing",closing)
    cv2.waitKey(0)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    cv2.imshow("or_image",or_image)
    cv2.waitKey(0)
    return or_image

file_path='bbbill1.jpg'
result=process_image_for_ocr(file_path)
#cv2.imshow("result",result)
#cv2.waitKey(0)
