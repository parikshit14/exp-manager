import cv2
import numpy as np


def get_preprocess(img_path):
    # Read image using opencv
    img = cv2.imread(img_path)
    # Rescale the image, if needed.
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # Converting to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(img, 0, 255,
    	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
    	angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
    	angle = -angle
    # rotate the image to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h),
    	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    #Removing Shadows
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)

    #Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)#increases the white region in the image
    img = cv2.erode(img, kernel, iterations=1) #erodes away the boundaries of foreground object

    #Apply blur to smooth out the edges
    #img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply threshold to get image with only b&w (binarization)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #Save the filtered image in the output directory
    cv2.imshow('result',img)
    cv2.waitKey(0)
    cv2.imwrite("bbbill_result2.jpg", img)
    #return result

get_preprocess('bbbill1.jpg')
