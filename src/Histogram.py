'''
Created on May 12, 2017
Calculate the color histogram from five beetles mask
Do not try to run this code on your computer, unless you change the name of 'Image4.jpg' to 'Image5.jpg'
and duplicate 'Image3.jpg' to 'Image4.jpg'

@author: Zhongchao
'''

import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np

def build_hist():
    # Initialize the color histogram
    hists = np.zeros(1000)
        
    for cnt in range(1, 6):
        imgname = '../tmp/Image' + str(cnt) + '.jpg'
        maskname = '../tmp/beetles' + str(cnt) + '.jpg'
        img = cv2.imread(imgname)
        
        # load the mask for each beetles, I force all the values to be either 0 or 255
        mask = cv2.imread(maskname)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask[mask<200] = 0
        mask[mask>199] = 255
        
        num = (mask > 0).sum()   
        
        # color histogram is rgb true color     
        hist = cv2.calcHist([img], [0, 1, 2], mask, [10, 10, 10], [0, 256, 0, 256,0, 256])
        hist = hist.ravel()
        
        # color histogram is normalized so that summation is one
        hist = hist / num
        hists += hist

    '''
    plt.plot(hists)
    plt.xlim([0, 8000])            
    plt.show()
    print
    '''
      
    # save to pickle file
    with open('../resource/hists.pkl', 'wb') as f:
        pickle.dump(hists, f)
     
        
if __name__ == '__main__':
    build_hist()