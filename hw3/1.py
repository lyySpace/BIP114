import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.morphology import disk, erosion, reconstruction
from skimage.util import img_as_bool
from skimage.measure import label, regionprops

img = io.imread("Fig1.tif") 
print(img)

# Convert to binary: 0 is background, 1 is disk
Disk_binary = img > 0.5
binary = img_as_bool(Disk_binary)
# Label connected components
labels = label(binary)
print("num objects:", labels.max()) 
radii = []

for region in regionprops(labels):
    area = region.area
    r = np.sqrt(area / np.pi)  
    radii.append(r)

radii_sorted = sorted(radii)
print(radii_sorted)

# Select structuring element with radius 18(the median radius)
r = 18
selem = disk(r)

''' Opening by Reconstruction '''
# Erosion
eroded = erosion(binary, selem)

# Reconstruction by dilation
reconstructed = reconstruction(eroded, binary, method='dilation')

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
ax = axes.ravel()

ax[0].imshow(binary, cmap='gray')
ax[0].set_title("Original (Binary)")
ax[0].axis('off')

ax[1].imshow(eroded, cmap='gray')
ax[1].set_title("After Erosion")
ax[1].axis('off')

ax[2].imshow(reconstructed, cmap='gray')
ax[2].set_title("Opening by Reconstruction")
ax[2].axis('off')

plt.tight_layout()
plt.savefig("Fig1_result.png", dpi=300)
