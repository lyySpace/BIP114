import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./data/Fig3-1.bmp', cv2.IMREAD_GRAYSCALE)

# Roberts Cross kernels
kx = np.array([[1, 0],
               [0, -1]], dtype=np.float32)
ky = np.array([[0, 1],
               [-1, 0]], dtype=np.float32)

# calculate gradients
gx = cv2.filter2D(img, cv2.CV_32F, kx)
gy = cv2.filter2D(img, cv2.CV_32F, ky)

# calculate gradient magnitude
grad = cv2.magnitude(gx, gy)
grad = np.clip(grad, 0, 255).astype(np.uint8)


plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(grad, cmap='gray')
plt.title('Roberts Cross Gradient')
plt.axis('off')

plt.tight_layout()
plt.savefig('./hw2_3.png', dpi=300)

