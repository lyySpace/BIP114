
import cv2, numpy as np, matplotlib.pyplot as plt

# (a): Raw image
a = cv2.imread('./data/Fig4-1.bmp', cv2.IMREAD_GRAYSCALE)

# (b): Laplacian of (a)
b = cv2.Laplacian(a, cv2.CV_32F, ksize=3)        
b_norm = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
b_norm = b_norm.astype(np.uint8)


# (c): Sharpened image obtained by adding (a) and (b)
c = cv2.convertScaleAbs(a.astype(np.float32) - b)

# (d): Sobel gradient of (a)
gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
d = cv2.convertScaleAbs(cv2.magnitude(gx, gy))

# (e): Smoothed with a 5*5 averaging filter of (d)
e = cv2.blur(d, (5, 5), borderType=cv2.BORDER_REFLECT)

# (f): Mask image formed by the product of (c) and (e)
f = cv2.convertScaleAbs((c.astype(np.float32) * e.astype(np.float32)) / 255.0)

# (g): Sharpened image obtained by the sum of (a) and (f)
g = cv2.add(a, f)

# (h): Applying a power-law transformation to (g), gamma = 0.5
h = (np.power(g.astype(np.float32)/255.0, 0.5) * 255.0 + 0.5).astype(np.uint8)


titles = ['(a) Original','(b) Laplacian','(c) a - b','(d) Sobel |∇f|',
          '(e) Smooth(d)','(f) c * e','(g) a + f','(h) Power-law γ=0.5']
imgs   = [a, b_norm, c, d, e, f, g, h]

plt.figure(figsize=(4, 12))
for i,(im,ti) in enumerate(zip(imgs, titles), 1):
    plt.subplot(4,2,i); plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.title(ti); plt.axis('off')
plt.tight_layout()
plt.savefig('./hw2_4.png', dpi=300)

