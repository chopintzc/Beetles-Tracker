'''
Created on May 14, 2017
This is an experiment for the 2 dimensional Fourier Analysis method, maybe promising to advance current methods
Not used in the real analysis

@author: Zhongchao
'''

import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np

if __name__ == '__main__':    
    img1 = cv2.imread('../tmp/beetles2.jpg')
    img2 = cv2.imread('../tmp/test2.jpg')
    
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    img2 = np.fft.fft2(img2)
    img2 = np.fft.fftshift(img2)
    mag2 = 20*np.log(np.abs(img2))
    

    plt.subplot(221), plt.imshow(mag2, cmap='gray')
    plt.show()
    
    print 