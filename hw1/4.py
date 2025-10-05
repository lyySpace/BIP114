import cv2
import numpy as np
import os

input_path  = "./Fig3.GIF"           
output_fold = "results_hw1_4"    
os.makedirs(output_fold, exist_ok=True)

img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12)) 
enh   = clahe.apply(gray)
cv2.imwrite(f"{output_fold}/clahe.png", enh)

# Gaussian adaptive thresholding
block = 41 # must be odd   
C     = 10 

bin_gauss = cv2.adaptiveThreshold(enh, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,
                                  block, C)
cv2.imwrite(f"{output_fold}/bin_gauss.png", bin_gauss)

