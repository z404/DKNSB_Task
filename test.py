# perform ocr on the image

import cv2
import numpy as np
import pytesseract
from PIL import Image

# read the image
img = cv2.imread('./images/result_Page_1.jpg')

# convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# save the thresholded image
# cv2.imwrite('./images/thresh.jpg', thresh)

# create rectangular kernel
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

# apply dilation on the threshold image
dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)

# save the dilated image
# cv2.imwrite('./images/dilation.jpg', dilation)

# find contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# create a copy of original image
im2 = img.copy()

# draw contours on the copy of original image
text = cv2.drawContours(im2, contours, -1, (0, 255, 0), 3)

# save the image with contours
# cv2.imwrite('./images/text.jpg', text)

# sort contours based on their area keeping minimum required area as '40' (anything smaller than this will not be considered)
contours = sorted(contours, key = cv2.contourArea, reverse = True)

# generate ocr for each contour
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Drawing a rectangle on copied image
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Cropping the text block for giving input to OCR
    cropped = im2[y:y + h, x:x + w]
    #save the cropped image
    # cv2.imwrite('./images/cropped.jpg', cropped)

    # Open the file in append mode
    # file = open("recognized.txt", "a")

    # # Apply OCR on the cropped image
    text = pytesseract.image_to_string(cropped, config='--psm 11')
    print(text)
    # # Appending the text into file
    # file.write(text)
    # file.write("\n")

    # # Close the file
    # file.close()
