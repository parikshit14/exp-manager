from pdf2image import convert_from_path

img = convert_from_path('invoice3.pdf')
img[0].save('invoice3.jpg')
