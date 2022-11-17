import cv2

from skimage.feature import match_template
from skimage.color import rgb2gray

img1 = cv2.imread('./images/result_Page_1.jpg')
img2 = cv2.imread('./symbols/9.png')[:,:,:3]

# make img 1 a square by increasing the height
img1 = cv2.resize(img1, (img1.shape[1], img1.shape[1]))

# save the image
cv2.imwrite('./res/resize.jpg', img1)

# match the images
result = match_template(rgb2gray(img1), rgb2gray(img2))

# get the best match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(max_val)

# draw a rectangle around the matched region
cv2.rectangle(img1, max_loc, (max_loc[0] + img2.shape[1], max_loc[1] + img2.shape[0]), (0, 255, 0), 2)

# show the result
cv2.imwrite('./res/result_Page_x2.jpg', img1)