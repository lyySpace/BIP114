import cv2
import numpy as np

input_path  = "./Fig2.bmp" 
img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

h, w = img.shape[:2]
cx, cy = (w - 1) / 2.0, (h - 1) / 2.0   # center

angles = [15, 30, 45, 60, 90]           
for ang in angles:
    M = cv2.getRotationMatrix2D((cx, cy), -ang, 1.0) # clockwise
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    cv2.imwrite(f"Fig2_rot_{ang}deg.bmp", out)

