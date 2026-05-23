import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

class Image():
    def __init__(self,imgpath):
        self.imgpath = imgpath

    def ImageReader(self):
        img = cv2.imread(self.imgpath)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        walldata = np.array([[],[]])
        actordata = []
        goaldata = []
        for y in range(0,10):
            for x in range(0,10):
                if np.array_equal(imgRGB[x,y],[0,0,0]) == True:
                    walldata = np.append(walldata, np.array([[y],[x]]), axis=1)
                elif np.array_equal(imgRGB[x,y],[0,255,0]) == True:
                    actordata = [y * 50,x * 50]
                elif np.array_equal(imgRGB[x,y],[255,0,0]) == True:
                    goaldata = [y * 50,x * 50]
        return walldata, actordata, goaldata

