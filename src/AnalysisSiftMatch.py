'''
Created on May 13, 2017
This is an experiment for the Sobel gradient and tuning curve measurement
Not used in the real analysis

@author: Zhongchao
'''

import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np

LEN = 36

def cal_CV(hist):
    max_idx = hist.argmax()
    if max_idx < LEN/2:
        min_idx = max_idx + LEN/2
    else:
        min_idx = max_idx - LEN/2
    if max_idx < LEN/4:
        l_idx = max_idx + LEN*3/4
    else:
        l_idx = max_idx - LEN/4
    if max < LEN*3/4:
        r_idx = max_idx + LEN/4
    else:
        r_idx = max_idx - LEN*3/4
    
    max_val = hist[max_idx] + hist[min_idx]
    min_val = hist[l_idx] + hist[r_idx]
    return (max_val-min_val)/(max_val+min_val)

if __name__ == '__main__':    
    img1 = cv2.imread('template1.jpg')
    img2 = cv2.imread('test2.jpg')
    
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gx1 = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=1)
    gy1 = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=1)
    gx2 = cv2.Sobel(img2, cv2.CV_32F, 1, 0, ksize=1)
    gy2 = cv2.Sobel(img2, cv2.CV_32F, 0, 1, ksize=1)
    
    mag1, angle1 = cv2.cartToPolar(gx1, gy1, angleInDegrees=True)
    mag2, angle2 = cv2.cartToPolar(gx2, gy2, angleInDegrees=True)
    
    gx1 = cv2.Sobel(angle1, cv2.CV_32F, 1, 0, ksize=1)
    gy1 = cv2.Sobel(angle1, cv2.CV_32F, 0, 1, ksize=1)
    gx2 = cv2.Sobel(angle2, cv2.CV_32F, 1, 0, ksize=1)
    gy2 = cv2.Sobel(angle2, cv2.CV_32F, 0, 1, ksize=1)
    
    mag1, angle1 = cv2.cartToPolar(gx1, gy1, angleInDegrees=True)
    mag2, angle2 = cv2.cartToPolar(gx2, gy2, angleInDegrees=True)
    
    mag1 = cv2.cvtColor(mag1,cv2.COLOR_BGR2GRAY)
    angle1 = cv2.cvtColor(angle1,cv2.COLOR_BGR2GRAY)
    
    mag2 = cv2.cvtColor(mag2,cv2.COLOR_BGR2GRAY)
    angle2 = cv2.cvtColor(angle2,cv2.COLOR_BGR2GRAY)
    
    histograms1 = cv2.calcHist([angle1], [0], None, [LEN], [0, 360])
    histograms1 = histograms1.ravel()
    hist1 = histograms1[0:LEN/2] + histograms1[LEN/2:]
    cv1 = cal_CV(histograms1)

    histograms2 = cv2.calcHist([angle2], [0], None, [LEN], [0, 360])
    histograms2 = histograms2.ravel()
    hist2 = histograms2[0:LEN/2] + histograms2[LEN/2:]
    cv2 = cal_CV(histograms2)
    
    print cv1, cv2
    
    #val = np.correlate(hist1, hist2)
    #print max(val)
    
    plt.subplot(221), plt.imshow(mag1, cmap='gray')
    plt.subplot(222), plt.imshow(angle2, cmap='gray')
    plt.subplot(223), plt.plot(histograms1)
    plt.subplot(224), plt.plot(histograms2)
    plt.show()
        
    print
