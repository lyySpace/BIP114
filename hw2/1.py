"""
Histogram tools:
(a) Compute and plot histogram(s) of an image (grayscale or color).
(b) Implement histogram equalization (grayscale; and Y channel for color images).

Usage:
    python hist_eq.py Fig1-1.png Fig1-2.jpg Fig1-3.tif Fig1-4.bmp

Outputs (per image):
    <name>_gray.png                 # grayscale version used for HE (if color)
    <name>_equalized.png            # histogram equalized result
    <name>_hist_before.png          # histogram before HE
    <name>_hist_after.png           # histogram after HE
    <name>_report.json              # basic stats

Notes:
- For color images, we equalize the Y (luma) channel in YCrCb and convert back to BGR/RGB-like.
- For grayscale images, we apply classic 256-bin HE using CDF.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

def is_grayscale(img: np.ndarray) -> bool:
    return (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)

def to_grayscale(img: np.ndarray) -> np.ndarray:
    if is_grayscale(img):
        if img.ndim == 3:
            return img[..., 0]
        return img
    # OpenCV loads as BGR by default
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def compute_histogram(img_gray: np.ndarray, bins: int = 256):
    # Assume 8-bit image [0,255]
    hist = np.bincount(img_gray.ravel(), minlength=256)[:bins]
    bin_edges = np.arange(bins+1)
    return hist, bin_edges

def equalize_histogram_gray(img_gray: np.ndarray):
    # Manual HE to be explicit
    flat = img_gray.ravel()
    hist = np.bincount(flat, minlength=256)
    cdf = hist.cumsum()
    # Mask zeros to avoid flat areas mapping to negative
    cdf_masked = np.ma.masked_equal(cdf, 0)
    # Normalize CDF to [0,255]
    cdf_min = cdf_masked.min()
    cdf_max = cdf_masked.max()
    cdf_norm = (cdf_masked - cdf_min) * 255 / (cdf_max - cdf_min)
    cdf_filled = np.ma.filled(cdf_norm, 0).astype('uint8')
    out = cdf_filled[flat].reshape(img_gray.shape)
    return out

def equalize_histogram_color_bgr(img_bgr: np.ndarray):
    # Equalize Y channel in YCrCb (keeps chroma; changes brightness/contrast)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    Y_eq = equalize_histogram_gray(Y)
    ycrcb_eq = cv2.merge([Y_eq, Cr, Cb])
    bgr_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    return bgr_eq, Y, Y_eq

def plot_histogram(img_gray: np.ndarray, title: str, out_path: Path):
    hist, bins = compute_histogram(img_gray, bins=256)
    plt.figure()
    plt.bar(bins[:-1], hist, width=1.0)  # default colors as required
    plt.xlim(0, 255)
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=144)
    plt.close()

def process_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARN] Cannot read: {path}")
        return

    # Convert 16-bit to 8-bit if needed (linear rescale to [0,255])
    if img.dtype == np.uint16:
        maxv = np.max(img) if np.max(img) > 0 else 65535
        img8 = np.clip((img.astype(np.float64) / maxv) * 255.0, 0, 255).astype(np.uint8)
        img = img8

    base = path.with_suffix('')
    out_report = base.as_posix() + "_report.json"

    if is_grayscale(img):
        gray = to_grayscale(img)
        eq = equalize_histogram_gray(gray)
        cv2.imwrite(base.as_posix() + "_gray.png", gray)
        cv2.imwrite(base.as_posix() + "_equalized.png", eq)
        plot_histogram(gray, f"{path.name} - Histogram (before)", Path(base.as_posix() + "_hist_before.png"))
        plot_histogram(eq, f"{path.name} - Histogram (after HE)", Path(base.as_posix() + "_hist_after.png"))
        stats = {
            "file": path.name,
            "mode": "grayscale",
            "before_min": int(gray.min()),
            "before_max": int(gray.max()),
            "after_min": int(eq.min()),
            "after_max": int(eq.max())
        }
    else:
        # Color path
        eq_bgr, Y, Y_eq = equalize_histogram_color_bgr(img)
        cv2.imwrite(base.as_posix() + "_equalized.png", eq_bgr)
        cv2.imwrite(base.as_posix() + "_gray.png", Y)  # luminance used
        plot_histogram(Y, f"{path.name} - Y Histogram (before)", Path(base.as_posix() + "_hist_before.png"))
        plot_histogram(Y_eq, f"{path.name} - Y Histogram (after HE)", Path(base.as_posix() + "_hist_after.png"))
        stats = {
            "file": path.name,
            "mode": "color(YCrCb-Y equalized)",
            "before_min": int(Y.min()),
            "before_max": int(Y.max()),
            "after_min": int(Y_eq.min()),
            "after_max": int(Y_eq.max())
        }

    with open(out_report, "w", encoding="utf-8") as f:
        import json
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[OK] Processed {path.name}")
    print(f"  -> Saved: {base.name}_equalized.png, {base.name}_hist_before.png, {base.name}_hist_after.png, {base.name}_report.json")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Example:\n  python hist_eq.py Fig1-1.png Fig1-2.jpg Fig1-3.tif Fig1-4.bmp")
        sys.exit(0)
    for p in sys.argv[1:]:
        process_image(Path(p))

if __name__ == "__main__":
    main()
