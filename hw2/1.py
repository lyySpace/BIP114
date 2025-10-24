import numpy as np
import cv2
import matplotlib.pyplot as plt

def Histogram(img):
    hist = np.zeros(256) # vector 

    for rows in img: # a row, nparray
        for element in rows:
            hist[element] += 1 

    # Probability Density Function (PDF)
    PDF = hist/hist.sum() # Normalized into [0,1] 

    return hist, PDF # vector 

'''Cumulative Distribution Function(CDF)'''
def CDF(img):
    cdf = np.zeros(256) # vector 

    _, pdf = Histogram(img)

    for idx, element in enumerate(pdf):
        if idx == 0:
            cdf[idx] += element
        else:
            cdf[idx] += (element + cdf[idx-1]) # 累加
    
    return cdf # vector 

'''Histogram Equalization'''
def HistogramEqual(img_input,L):
    img_out = np.zeros(shape=img_input.shape) 
    cdf = CDF(img_input) # vector 

    for rows, element_in_col in enumerate(img_input):
        for cols, element in enumerate(element_in_col):
            img_out[rows, cols] = L*cdf[element] # The transform function
    return img_out

paths = ['./data/Fig1-1.bmp', './data/Fig1-2.bmp', './data/Fig1-3.bmp', './data/Fig1-4.bmp']
imgs = []
Hists = []
PDFs = []
His_Eqs = []
Eqs_hists = []

for p in paths:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    imgs.append(img)
    hist, pdf = Histogram(img)
    Hists.append(hist)
    PDFs.append(pdf)
    his_eq = HistogramEqual(img, 255)
    His_Eqs.append(his_eq)
    eq_hist, _ = Histogram(his_eq.astype(np.uint8))
    Eqs_hists.append(eq_hist)
    


fig, axes = plt.subplots(4, 5, figsize=(18, 16))

for i in range(4):
    axes[i, 0].imshow(imgs[i], cmap='gray', vmin=0, vmax=255)
    axes[i, 0].set_title(f'Fig1-{i+1}')
    axes[i, 0].axis('off')

    axes[i, 1].bar(range(len(Hists[i])), Hists[i], color='green', width=1.0)
    axes[i, 1].set_title('Histogram')
    axes[i, 1].set_xlim(0, 255)

    axes[i, 2].bar(range(len(PDFs[i])), PDFs[i], color='green', width=1.0)
    axes[i, 2].set_title('PDF')
    axes[i, 2].set_xlim(0, 255)

    axes[i, 3].imshow(His_Eqs[i], cmap='gray', vmin=0, vmax=255)
    axes[i, 3].set_title('Histogram Equalization Result')
    axes[i, 3].axis('off')

    axes[i, 4].bar(range(len(Eqs_hists[i])), Eqs_hists[i], color='green', width=1.0)
    axes[i, 4].set_title('Equalized Histogram')
    axes[i, 4].set_xlim(0, 255)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('./hw2_1.png', dpi=300)
