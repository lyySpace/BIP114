import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
import matplotlib.pyplot as plt
from skimage.morphology import square, erosion
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.util import img_as_float
from skimage.color import gray2rgb

img = io.imread("Fig2.gif")
print(img.shape)
img = img[0]   # Tiff could have multiple pages

if img.ndim == 3:
    img_gray = rgb2gray(img)
else:
    img_gray = img_as_float(img)

# Otsu's thresholding
t = threshold_otsu(img_gray)
binary = img_gray > t

X = binary

# structuring element, SE
selem = square(3) # 3x3 pixels square

# Erosion
X_eroded = erosion(X, selem)

# edge = X - (~X_eroded)  
edge = X & (~X_eroded) 

# radius search range (pixel units)
radii = np.arange(20, 61, 2)

# Hough circle transform
hough_res = hough_circle(edge, radii)

# Retrieve the strongest peak (the most likely circle)
accums, cx, cy, radii_found = hough_circle_peaks(hough_res, radii,
                                                 total_num_peaks=1)

# Drawing the circle perimeter on the original image
img_rgb = gray2rgb(img_gray)
for center_y, center_x, radius in zip(cy, cx, radii_found):
    rr, cc = circle_perimeter(center_y, center_x, radius)
    img_rgb[rr, cc] = (0, 100, 0)  

# Plotting
plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1); plt.imshow(edge, cmap='gray'); plt.title("Edges")
plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(img_gray, cmap='gray'); plt.title("Original gray")
plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(img_rgb); plt.title("Circle detected")
plt.axis('off')
plt.tight_layout()
plt.savefig("Fig2c_result.png", dpi=300)

print("Detected circle center(s) (x, y) and radius:", list(zip(cx, cy, radii_found)))
