import cv2
import os
import numpy as np

filepath = "C:\\Users\\seoir\\git\\replicating_old_papers\\cell_data"

n = 5

file_orig = filepath+"\\Original\\"+str(n)+".jpg"
label = filepath + "\\Manual\\"+str(n)+"s.jpg"
#print(os.listdir(filepath))

img = cv2.imread(file_orig)
img_label = cv2.imread(label,cv2.IMREAD_GRAYSCALE)

# Draw in the labels
# note that the label are where it is white (255), but there is some noise
# where the mask is value 1, should be removed prior to training
img[img_label == 255] = (0,0,255)
window_name = 'demo'


# Find the outer extents
height, width = img_label.shape
min_x, min_y = width, height
max_x = max_y = 0

cnts, _ = cv2.findContours(img_label, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for cnt in cnts:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    min_x = min(min_x,x)
    min_y = min(min_y,y)
    max_x = max(max_x,x+w)
    max_y = max(max_y,y+h)

cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)





cv2.imshow(window_name,img)



cv2.waitKey(0)

cv2.imwrite("images\\overlay_" + str(n) + ".png",img)

cv2.destroyAllWindows()


