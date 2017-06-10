'''
Created on May 13, 2017
Calculate correlation map

@author: Zhongchao
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math


length = 20     # size of sliding window 
step = 10       # step size of sliding window
LEN = 36        # 0-360 spatial directions are divided into 36 bins


class Target(object):    
    def __init__(self): 
        # load saved pickle file for beetles color histogram
        with open('hists.pkl', 'rb') as f:
            self.hists = pickle.load(f) 
        
        # load the image to process    
        self.frame = cv2.imread('image1.jpg')      
        
        # calculate the spatial gradient for the whole image
        self.frame = np.float32(self.frame)
        gx = cv2.Sobel(self.frame, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(self.frame, cv2.CV_32F, 0, 1, ksize=1)
        
        # transform the gradient into polar space
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        # calculate the spatial gradient for the gradient of image
        gx = cv2.Sobel(angle, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(angle, cv2.CV_32F, 0, 1, ksize=1)
        
        # transform the gradient of gradient into polar space
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        # convert gradient of gradient into gray scale (0-360)
        self.angle = cv2.cvtColor(angle,cv2.COLOR_BGR2GRAY)
        
        
    '''
        calculate circular variance for the oriented spatial pattern
    '''    
    def cal_CV(self, hist):
        # max_idx is the optimal spatial direction
        max_idx = hist.argmax()
        
        # min_idx is the opposite of optimal spatial direction
        if max_idx < LEN/2:
            min_idx = max_idx + LEN/2
        else:
            min_idx = max_idx - LEN/2
            
        # l_idx is the left orthogonal direction of optimal direction
        if max_idx < LEN/4:
            l_idx = max_idx + LEN*3/4
        else:
            l_idx = max_idx - LEN/4
            
        # r_idx is the right orthogonal direction of optimal direction
        if max < LEN*3/4:
            r_idx = max_idx + LEN/4
        else:
            r_idx = max_idx - LEN*3/4
        
        max_val = hist[max_idx] + hist[min_idx]
        min_val = hist[l_idx] + hist[r_idx]
        
        # circular variance is calculated as (best_orientation - orthogonal orientation) / (best_orientation + orthogonal orientation)
        return (max_val-min_val)/(max_val+min_val)
    
    
    '''
        calculate the spatial orientation tuning curve
    '''
    def cal_gradient(self, ystart, yend, xstart, xend):
        img = self.angle[ystart:yend, xstart:xend]
        histograms = cv2.calcHist([img], [0], None, [LEN], [0, 360])
        histograms = histograms.ravel()
        
        # call cal_CV to calculate circular variance for the tuning curve
        cv = self.cal_CV(histograms)
        return cv
    
    
    '''
        calculate the correlation coefficient between query image and template bettles image
    '''
    def cal_hists(self, ystart, yend, xstart, xend):
        img = self.frame[ystart:yend, xstart:xend, :]
        shape = img.shape
        num = shape[0] * shape[1]
        
        # the true color histogram is calculated for the query image patch
        histograms = cv2.calcHist([img], [0, 1, 2], None, [10, 10, 10], [0, 256, 0, 256,0, 256])
        histograms = histograms.ravel()
        histograms = histograms / num
        
        # calculate the correlation coefficient
        val = np.corrcoef(self.hists, histograms)

        return val[0,1]
    
    
    def run(self):          
        # calculate the size of query image and initialize the correlation map
        shape = self.frame.shape
        xlen = int(math.ceil((shape[1]-length) / step)) + 1
        ylen = int(math.ceil((shape[0]-length) / step)) + 1
        corrmap = [[x for x in range(xlen)] for y in range(ylen)]
        
        '''
            use a sliding window to sample a small region of query image
            the correlation value for each sliding window is the multiplication between correlation coefficient
            and circular variance of spatial tuning curve
        '''
        for x in range(0, xlen):
            for y in range(0, ylen):
                # if sliding window is inside query image
                if y*step+length <= shape[0] and x*step+length <= shape[1]:
                    corrmap[y][x] = self.cal_hists(y*step, y*step+length, x*step, x*step+length)
                    corrmap[y][x] *= self.cal_gradient(y*step, y*step+length, x*step, x*step+length)
                # if sliding window is to the right of query image    
                elif y*step+length > shape[0] and x*step+length <= shape[1]:
                    corrmap[y][x] = self.cal_hists(y*step, shape[0], x*step, x*step+length)
                    corrmap[y][x] *= self.cal_gradient(y*step, shape[0], x*step, x*step+length)
                # if sliding window is to the bottom of query image
                elif y*step+length <= shape[0] and x*step+length > shape[1]:
                    corrmap[y][x] = self.cal_hists(y*step, y*step+length, x*step, shape[1])
                    corrmap[y][x] *= self.cal_gradient(y*step, y*step+length, x*step, shape[1])
                # if sliding window is to the bottom right of query image
                else:
                    corrmap[y][x] = self.cal_hists(y*step, shape[0], x*step, shape[1])
                    corrmap[y][x] *= self.cal_gradient(y*step, shape[0], x*step, shape[1])
        
        # save correlation map into pickle fie
        with open('test1.pkl', 'wb') as f:
            pickle.dump(corrmap, f)
        
        '''
        b,g,r = cv2.split(frame)       # get b,g,r
        rgb_img = cv2.merge([r,g,b])     # switch it to rgb
        
        plt.imshow(rgb_img)
        plt.xticks([]), plt.yticks([])
        plt.show()
        '''
        

if __name__ == '__main__':
    target = Target()
    target.run()
