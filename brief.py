# source https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html

import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('/home/krutika/Documents/Feature_Descriptors/room.jpeg')

orb = cv2.ORB_create() # create object 

key_points = orb.detect(img, None) # Find keypoints

key_points, dst = orb.compute(img,key_points) # Find descriptors
img2 = cv2.drawKeypoints(img, key_points,None, color=(0,255,0), flags=0) 
plt.imshow(img2), plt.show() # Plot the image with keypoints.