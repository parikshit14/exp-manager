#file to extract the text from the image using tesseract


from pytesseract import *
from PIL import Image
import cv2
import pandas as pd


img = cv2.imread("invoice3.jpg")    #reading the file
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #changing the color from BGR to grayscale
#img=Image.open('file.jpg')
df = pytesseract.image_to_data(img, lang='eng', output_type=Output.DATAFRAME )    #extracting the text in dataframe format
df.to_csv('files.csv')
extracted_text=pytesseract.image_to_string(img)
file = open("invoice_txt.txt","w")
file.writelines(extracted_text)
file.close()
