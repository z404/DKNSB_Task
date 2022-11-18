import cv2
import numpy as np
from pdf2image import convert_from_path

images = []
pages = convert_from_path("Label.pdf", 500)
for page in pages:
    images.append(np.array(page))

images[0] = cv2.resize(images[0], (1436, 761))

cv2.imshow("image", images[0])
cv2.waitKey(0)