import cv2
import numpy as np

img1 = cv2.imread('./images/result_Page_2.jpg')

print(img1.shape)

# make img 1 a square by increasing the height
img1 = cv2.resize(img1, (img1.shape[1], img1.shape[1]))

height, width = 175, 210
template_rows = [800, 800+height]
template_cols = [150, 150+width, 150+2*width, 150+3*width]

# read symols directory
symbols = []
for i in range(1, 10):
    symbols.append(cv2.imread('./symbols/{}.png'.format(i))[:,:,:3])

#fix first template by making it a square
symbols[0] = cv2.resize(symbols[0], (width - 30, height - 30))
cv2.imshow('symbol', symbols[0])

for row_num, row_val in enumerate(template_rows):
    for col_num, col_val in enumerate(template_cols):
        template = img1[row_val - 75:row_val+height + 75, col_val - 75:col_val+width + 75]
        template_crop = template[75:-75, 75:-75]
        if np.mean(template_crop) > 245:
            continue
        
        max_temp_score, max_temp = 0, 0
        for symbol_num, symbol in enumerate(symbols):
            result = cv2.matchTemplate(template, symbol, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val > max_temp_score:
                max_temp_score = max_val
                max_temp = symbol_num
            print('row: {}, col: {}, symbol: {}, score: {}'.format(row_num, col_num, symbol_num, max_val))
        print('row: {}, col: {}, max_temp: {}'.format(row_num, col_num, max_temp + 1))


ce_check = img1[220:400, 1050:-125]
print(np.mean(ce_check))
cv2.imwrite('./res/ce_check.jpg', ce_check)