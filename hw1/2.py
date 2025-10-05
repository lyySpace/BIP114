import cv2, numpy as np

input_path  = "./Fig1.bmp"         
img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img

N = int(input("Enter gray levels (power of 2, e.g. 2/4/8/.../256): ").strip())
if not (2 <= N <= 256 and (N & (N-1)) == 0):
    raise ValueError("N must be the power of 2 and in range [2, 256]")

# Quantization
step = 256 // N
q = (gray // step) * step
q = q.astype(np.uint8)

output_path = f"Fig1_quantized_L{N}.bmp"
cv2.imwrite(output_path, q)


