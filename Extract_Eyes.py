import numpy as np
import cv2
import os
import shutil
import math
import matplotlib.pyplot as plt
import skimage
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from PIL import Image
import csv

face_cascade = cv2.CascadeClassifier('face.xml')
eye_cascade = cv2.CascadeClassifier('eye.xml')

path = "D://MajorProject//MeghaSharma//";
files = os.listdir(path);

print(files)

for file in files:
    img = cv2.imread(file);

    #print(file);

    img = np.full((100,80,3), 12, np.uint8)
    img = np.array(img, dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    for (x,y,w,h) in faces:
        print("inside")
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        r_gray = gray[y:y+h, x:x+w]
        r_color = img[y:y+h, x:x+w]
            
        eyes = eye_cascade.detectMultiScale(r_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(r_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        count = 1;
        for (ex,ey,ew,eh) in eyes[1:]:
            x1 = ex;
            y1 = ey;
            x2 = ex+ew;
            y2 = ey+eh;

            imgEye = r_color[y1:y2,x1:x2];
            print("here")
            skimage.io.imsave("D://MajorProject//MeghaSharmaEyes//"+file,imgEye);
            count=count+1;

    #cv2.imshow('img',img);
    #cv2.imwrite('D:\\MajorProject\\Detected_Eyes.png',img);
