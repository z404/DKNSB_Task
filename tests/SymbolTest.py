from skimage.io import imread
from skimage.feature import match_template
from skimage.color import rgb2gray
import numpy as np

image = imread('./images/result_Page_2.jpg')
template = imread('./symbols/4.png')[:,:,:3]

image_gray = rgb2gray(image)
template_gray = rgb2gray(template)

result = match_template(image_gray, template_gray)

# check if the template is found
if np.max(result) > 0.8:
    print('Found')
else:
    print('Not found')

ij = np.unravel_index(np.argmax(result), result.shape)
x, y = ij[::-1]

print(x, y)

import cv2

img = cv2.imread('./images/result_Page_2.jpg')
# draw a rectangle around the matched region
cv2.rectangle(img, (x, y), (x + template.shape[1], y + template.shape[0]), (0, 255, 0), 2)
cv2.imwrite('./images/result_Page_x2.jpg', img)