import matplotlib.pyplot as plt
from skimage.morphology import square, erosion, dilation
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.util import img_as_float

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

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1); plt.imshow(X, cmap='gray'); plt.title("Binary")
plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(X_eroded, cmap='gray'); plt.title("Eroded")
plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(edge, cmap='gray'); plt.title("Edge (X - erode)")
plt.axis('off')
plt.tight_layout()
plt.savefig("Fig2b_result.png", dpi=300)
