import matplotlib.pyplot as plt
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

# Plotting
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1); plt.imshow(img_gray, cmap='gray'); plt.title("Gray")
plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(binary, cmap='gray'); plt.title("Binary")
plt.axis('off')
plt.tight_layout()
plt.savefig("Fig2a_result.png", dpi=300)
