# import the necessary packages
# from cv2 import pyimagesearch
# from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from pytesseract import *
from PIL import Image

def set_image_dpi(file_path):
	im = Image.open(file_path)
	length_x, width_y = im.size
	print("IN THE PRINTER",float(100 / length_x))
	factor = min(1, float(500 / length_x))
	size = int(factor * length_x), int(factor * width_y)
	im_resized = im.resize(size, Image.ANTIALIAS)
	# temp_file = tempfile.NamedTemporaryFile(delete=False,   suffix='.jpg')
	# temp_filename = temp_file.name
	out='kuchbhi.jpg'
	im_resized.save('kuchbhi.jpg', dpi=(300, 300))
	cv2.imshow("in_dpi",cv2.imread(out))
	cv2.waitKey(0)
	# im_resized.show()
	return out

def image_smoothening(img):
    cv2.imshow("before smooth",img)
    cv2.waitKey(0)
    """ret1, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)"""
    height,width=img.shape
    gaussian_filter_img = cv2.GaussianBlur(img,(1,1),0)
    cv2.imshow("smooth",gaussian_filter_img)
    cv2.waitKey(0)
    return gaussian_filter_img

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())
img=set_image_dpi(args["image"]) #returns edited image name/location
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(img)
ratio = image.shape[0] / 500.0
orig = image.copy() #storing the original image
image = imutils.resize(image, height = 500)
# convert the image to grayscale, blur it, and find edges
# in the image
# gray = image_resize(image,300,700)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 75, 200)
# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image to going in canny", image)
cv2.imshow("canny edge detections", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)  #drawing bounding box/contours
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.imwrite("best_one.jpg",imutils.resize(warped, height = 650))
best_one=cv2.imread("best_one.jpg")
cv2.waitKey(0)
# smooth = cv2.GaussianBlur(warped,(5,5),0)#image_smoothening(warped)
enlarged=image_resize(warped,600,600)
cv2.imshow("enlarged",enlarged)
cv2.waitKey(0)

smooth = cv2.GaussianBlur(enlarged,(5,5),0)#image_smoothening(warped)
cv2.imwrite("result.jpg",enlarged)
img=set_image_dpi("result.jpg")
img=cv2.imread(img)
dst = cv2.detailEnhance(img, sigma_s=7, sigma_r=0.15)
cv2.imshow("last",dst)
cv2.waitKey(0)
# df = pytesseract.image_to_data(warped, lang='eng', output_type=Output.DATAFRAME )    #extracting the text in dataframe format
# # df.to_csv('files10.csv')
# text = target = pytesseract.image_to_string(warped, lang='eng')
#
# print(text)
results=pytesseract.image_to_data(smooth, output_type=Output.DICT)

# Then loop over each of the individual text
# localizations
for i in range(0, len(results["text"])):
	# extract the bounding box coordinates of the text region from
	# the current result
	x = results["left"][i]
	y = results["top"][i]
	w = results["width"][i]
	h = results["height"][i]
	# extract the OCR text itself along with the confidence of the
	# text localization
	text = results["text"][i]
	conf = int(results["conf"][i])
	if conf > 20:
		# display the confidence and text to our terminal
		# print("Confidence: {}".format(conf))
		# print("Text: {}".format(text))
		# print("")
		# strip out non-ASCII text so we can draw the text on the image
		# using OpenCV, then draw a bounding box around the text along
		# with the text itself
		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		cv2.rectangle(smooth, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(smooth, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
			1, (200, 0 ,0), 3)
# show the output image
cv2.imshow("Image", smooth)
cv2.waitKey(0)
cv2.imwrite("check.jpg",warped)
