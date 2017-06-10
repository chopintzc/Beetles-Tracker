'''
Created on May 14, 2017
Find out the region of interest and draw with red rectangle

@author: Zhongchao
'''

import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np


MIN_THRESH = 132    # 102 
MAX_THRESH = 143    # 148 143
AREA_THRESH = 2     # 2
length = 20         # size of sliding window 
step = 10           # step size of sliding window
rect = 50           # size of rectangle


if __name__ == '__main__':
    # image number to process
    file = 2
    
    # load the correlation map from pickle file
    with open('test'+str(file)+'.pkl', 'rb') as f:
        corrmap = pickle.load(f)
    
    
    plt.imshow(corrmap, cmap='gray', vmin=0, vmax=0.6)
    plt.show()
    print
    
        
    # load the original image    
    img = cv2.imread('Image'+str(file)+'.jpg')
    
    # convert correlation map into numpy array
    corrmap = [[x*255 for x in y] for y in corrmap]
    corrmap = np.array(corrmap)
    np.clip(corrmap, 0, 255, out=corrmap)
    
    # convert correlation map into uint8 type
    corrmap = corrmap.astype('uint8')
    
    # calculate the thresholded image and find contours for all the patches
    retval, thresh = cv2.threshold(corrmap, MIN_THRESH, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    roi = []
    
    # loop through each contour
    for contour in contours:
        # discard the contour if its area too small
        if len(contour) < AREA_THRESH:
            continue
        '''
            find out contour of interest if its maximal correlation value is over the threshold
            store the location of maximal value so that rectangle can be draw later
        '''
        flag = False
        max_val = MAX_THRESH
        for c in contour:
            if corrmap[c[0][1]][c[0][0]] > MAX_THRESH:
                flag = True
                if corrmap[c[0][1]][c[0][0]] > max_val:
                    max_val = corrmap[c[0][1]][c[0][0]]
                    x = c[0][0]
                    y = c[0][1]
        
        # if current contour fit our interest, draw the red rectangle around the center of ROI        
        if flag:
            roi.append(contour)
            x = x * step + length / 2 - rect / 2
            y = y * step + length / 2 - rect / 2
            cv2.rectangle(img, (x, y), (x + rect, y + rect), (0,0,255), 5)
    '''        
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    print
    '''
    
    # write the solution
    cv2.imwrite('solution'+str(file)+'.jpg', img)
    
    