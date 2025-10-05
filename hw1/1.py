import cv2
import numpy as np
from math import sqrt

input_path  = "./Fig1.bmp"         
output_path = "./Fig1_triangle.bmp"
top = (150, 150)                 
side = 200                        
gray = 255                        
thickness = 2                     

img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

x0, y0 = top
h = (sqrt(3)/2.0) * side
T = (int(round(x0)), int(round(y0)))
L = (int(round(x0 - side/2.0)), int(round(y0 + h)))
R = (int(round(x0 + side/2.0)), int(round(y0 + h)))
pts = np.array([T, L, R], dtype=np.int32)

cv2.polylines(img, [pts], isClosed=True, color=gray, thickness=thickness)
cv2.imwrite(output_path, img)

