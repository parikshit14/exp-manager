import cv2
import numpy as np
import utlis
from pytesseract import *

########################################################################
# webCamFeed = False
pathImage = "bbbill1.jpg"
# cap = cv2.VideoCapture(1)
# cap.set(10,160)
# heightImg = 1000
# widthImg  = 700
########################################################################

# #skew correction i.e perfectly alligning the image
# def deskew(image):
#     img = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
#     # Converting to gray scale
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     coords = np.column_stack(np.where(thresh > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#     # the `cv2.minAreaRect` function returns values in the
#     # range [-90, 0); as the rectangle rotates clockwise the
#     # returned angle trends to 0 -- in this special case we
#     # need to add 90 degrees to the angle
#     if angle < -45:
#     	angle = -(90 + angle)
#     # otherwise, just take the inverse of the angle to make
#     # it positive
#     else:
#     	angle = -angle
#     # rotate the image to deskew it
#     (h, w) = img.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # return img
# image=cv2.imread(pathImage)
# rotated=deskew(image)
# cv2.imshow("rotated",rotated)
# cv2.waitKey(0)
# cv2.imwrite("after-rotation.jpg",rotated)
# utlis.initializeTrackbars()
# count=0
# i=0
while True :

    # if webCamFeed:success, img = cap.read()
    # else:

    #reading image and calculating aspect ratio
    image = cv2.imread(pathImage)
    heightImg,widthImg = image.shape[:2]
    asp_ratio=float(heightImg)/float(widthImg)
    print("asp_ratio",asp_ratio)
    print(heightImg,widthImg)
    # img = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # Converting to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)
    cv2.waitKey(0)


    # skew correction
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("thresh",thresh)
    cv2.waitKey(0)
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
    center = (widthImg // 2, heightImg // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # cv2.imshow("M",M)
    # cv2.waitKey(0)
    img = cv2.warpAffine(image, M, (widthImg, heightImg), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imshow("rotated image",img)
    cv2.waitKey(0)


    # img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
    #ADDING BORDER TO OVERCOME INCOMPLETE IMAGE
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT) #adding border for incomplete image
    cv2.imshow("bordered image",img)
    cv2.waitKey(0)

    #1)BLURING 2)CANNY-EDGE-DETECTION(DETECTS MAIN OBJECTS AND TURN OTHERS BLACK)
    #3)DIALATING- CLEANING OR DILUTING OR MOTA KARNA
    #4)ERODING- THINING OR PATLA KARNA
    # imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(img, (5, 5), 1) # ADD GAUSSIAN BLUR
    # thres=utlis.valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur,10,50) # APPLY CANNY BLUR( provides options to change the threshold values)
    canned=imgThreshold
    cv2.imshow("canny image",imgThreshold)
    cv2.waitKey(0)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    cv2.imshow("eroded",imgDial)
    cv2.waitKey(0)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
    cv2.imshow("dilated",imgThreshold)
    cv2.waitKey(0)

    # APPLY DILATION (mota karna)
      # APPLY EROSION (patla karna)

    # imgThreshold = cv2.dilate(imgThreshold, kernel, iterations=3)
    ## FIND ALL COUNTOURS

    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    _, contours,_ = cv2.findContours(imgThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS
    resized=cv2.resize(imgContours,(heightImg//2,widthImg//2))
    cv2.imshow("contoured image",resized)
    cv2.waitKey(0)
    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = utlis.biggestContour(contours) # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        # print("biggest",biggest)



        biggest,flag=utlis.reorder(biggest)
        if flag == 1:
            for contour in contours:
                (x,y,w,h) = cv2.boundingRect(contour)
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 10)

            print("error!!")
            img=cv2.resize(img,(heightImg//2,widthImg//2))
            cv2.imshow("image for try",img)
            cv2.waitKey(0)
            break
        # out = img[ topx:bottomx+1,topy:bottomy+1]
        # cv2.imshow("sample cropping",out)
        # cv2.waitKey(0)
        # cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
        # imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
        print("after reordering",biggest)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        print("pts1",pts1)
        # new_wid=biggest[1][0][0]
        # new_hig=biggest[2][0][1]
        # print("a",new_wid,new_hig)
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        print("pts2",pts2)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        # cv2.imshow("imgWarpColored",imgWarpColored)
        # cv2.waitKey(0)
        # imgWarpColored1 = imgWarpColored[topx:bottomx+1,topy:bottomy+1]
        # cv2.imshow("imgWarpColored1",imgWarpColored1)
        # cv2.waitKey(0)
        # asp_ratio = new_hig/new_wid
        # print("new asp_ratio",asp_ratio)
        #REMOVE 20 PIXELS FORM EACH SIDE #removed-20 as it was cropping too much
        # crop_w=5
        # crop_h=int(asp_ratio*crop_w)
        imgWarpColored=imgWarpColored[10:imgWarpColored.shape[0] , 5:imgWarpColored.shape[1]]
        # imgWarpColored = cv2.resize(imgWarpColored,(new_wid,new_hig))
        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        # imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        kernel = np.array([[0, -1, 0],
                          [-1, 5,-1],
                          [0, -1, 0]])

        # Sharpen image
        image_sharp = cv2.filter2D(imgAdaptiveThre, -1, kernel)
        # image_sharp = cv2.fastNlMeansDenoisingMulti(image_sharp, 2, 1, None, 4, 7, 35)
        cv2.imshow("sharp",image_sharp)
        cv2.waitKey(0)
        imgAdaptiveThre=cv2.medianBlur(image_sharp,3)

        # Image Array for Display
        imageArray = ([img,gray,imgThreshold,imgContours],
                      [imgBigContour,imgWarpColored, imgWarpGray,imgAdaptiveThre])

    # else:
    #     print("error encountered, ABORTING!!")
    #     break
    #     imageArray = ([img,imgGray,imgThreshold,imgContours],
    #                   [imgBlank, imgBlank, imgBlank, imgBlank])

    # LABELS FOR DISPLAY
    lables = [["Original","Gray","Threshold","Contours"],
              ["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]

    # stackedImage = utlis.stackImages(imageArray,0.75,lables)
    # cv2.imshow("Result",stackedImage)
    cv2.imshow("best_one",imgAdaptiveThre)
    cv2.waitKey(0)


    results=pytesseract.image_to_data(imgAdaptiveThre, output_type=Output.DICT)
    extracted_text=pytesseract.image_to_string(imgAdaptiveThre)
    # target = pytesseract.image_to_string(imgAdaptiveThre, lang='eng')

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
    		cv2.rectangle(imgAdaptiveThre, (x, y), (x + w, y + h), (0, 255, 0), 2)
    		cv2.putText(imgAdaptiveThre, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,1, (200, 0 ,0), 3)
    # show the output image


    # print(results)
    # print(target)
    # File_object = open("sample.txt","w+")
    # File_object.write(text)
    # file = open("myfile.txt","w")
    # file.writelines(extracted_text)
    # file.close()
    # with open('myfile1.txt', 'a') as f:
    #     f.write(text)
    # print("text:",text)
    # print("target:",target)

    cv2.imshow("Image", imgAdaptiveThre)
    if cv2.waitKey(0):
        break


    # SAVE IMAGE WHEN 's' key is pressed
    # if cv2.waitKey(0) & 0xFF == ord('s'):
    #     cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgWarpColored)
    #     cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
    #                   (1100, 350), (0, 255, 0), cv2.FILLED)
    #     cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
    #                 cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
    #     cv2.imshow('Result', stackedImage)
    #     cv2.waitKey(0)
    #     count += 1
