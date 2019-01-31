import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('/home/krutika/Documents/Feature_Descriptors/room.jpeg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create() # create object 

kp = sift.detect(gray,None) # detect the key points

image = cv2.drawKeypoints(gray,kp,img) # draw the keypoints on the gray image

plt.imshow(image)
plt.show()