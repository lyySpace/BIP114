import cv2
import matplotlib.pyplot as plt

m_sizes = [3, 5, 9, 15, 35]

img = cv2.imread('./data/Fig2-1.bmp', cv2.IMREAD_GRAYSCALE)

results = [img]  
titles  = ['Original']
for m in m_sizes:
    out = cv2.blur(img, (m, m), borderType=cv2.BORDER_REFLECT)
    results.append(out)
    titles.append(f'm = {m}')

plt.figure(figsize=(6, 8))
for i, (res, title) in enumerate(zip(results, titles)):
    plt.subplot(3, 2, i + 1)
    plt.imshow(res, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.savefig('./hw2_2.png', dpi=300)
