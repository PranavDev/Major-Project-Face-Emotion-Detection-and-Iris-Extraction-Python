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
from skimage.filters import sobel
from pygame import mixer
from time import sleep
import pandas as pd
from PIL import Image
import csv
import random
import socket


testDataframe = []
cols=[]

def connect(message,emotion):
    s = socket.socket()
    s.connect(('192.168.2.11',40481))
    #while True:
    string = str(message)+"_"+str(emotion)
    s.send(string.encode());
        #if(str == "Bye" or str == "bye"):
        #    break
        #print ("Received Message:",s.recv(1024).decode())
    s.close()






def captureLiveImage():
    cap = cv2.VideoCapture(0)
    
    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('D:\\MajorProject\\Face.png',frame)
            break

    cap.release();
    cv2.destroyAllWindows();
    return;





def Extract_Eyes():
    face_cascade = cv2.CascadeClassifier('face.xml')
    eye_cascade = cv2.CascadeClassifier('eye.xml')

    path = "D:\\MajorProject\\Face.png";
    img = cv2.imread(path);
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
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

            cv2.imwrite('D:\\MajorProject\\IrisPreProcessing\\eye'+str(count)+'.png',imgEye);
            count=count+1;

    #cv2.imshow('img',img);
    cv2.imwrite('D:\\MajorProject\\Detected_Eyes.png',img);
    return;




def IrisPreprocessing():
    folder1 = "D://MajorProject//IrisPreProcessing1//";
    folder2 = "D://MajorProject//IrisProcessed//";
    files = os.listdir(folder1)
    
    for file in files :
        im1 = skimage.io.imread(folder1+file)
        #skimage.io.imsave(folder1+'original.bmp',im1);
        im2 = skimage.color.rgb2gray(im1)

        im2 = skimage.exposure.rescale_intensity(im2, out_range=(0,255))
        #skimage.io.imsave(folder1+'intensityRescale.bmp',im2);
        # Compute the Canny filter for two values of sigma
        edges = skimage.feature.canny(im2)
        edges1 = edges * 240;

        hough_radii = np.arange(45, 120, 2)
        hough_res = hough_circle(edges1, hough_radii)

        #skimage.io.imsave(folder1+'cannyEdges.bmp',edges1);
        # Select the most prominent 5 circles
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=1)

        x1 = (int)(cx-radii)
        x2 = (int)(cx+radii)
        y1 = (int)(cy-radii)
        y2 = (int)(cy+radii)
        im2 = edges1[y1:y2,x1:x2];
        #skimage.io.imsave(folder1+'irisExtraction.bmp',im2);
        m = im2.shape[0];
        n = im2.shape[1];

        #im3 = skimage.transform.rescale(im2,(160/m,160/n));
        im3 = im2;
        #skimage.io.imsave(folder1+'resizingImg.bmp',im3);
        im4 = np.zeros((110, 110), dtype=np.uint8);
        m = im3.shape[0];
        n = im3.shape[1];

        r = (int)(m/2)
        r2 = (int)(n/2)
        d = 0
        dist = 0.0
        for i in range(1,m,1):
            for j in range(1,n,1):
                dist = (i-r)*(i-r) + (j-r2)*(j-r2);
                dist = math.sqrt(dist)
                d = (int)(dist+1)
                if(d <= r):
                    im4[i,j] = im3[i,j];

                
        #skimage.io.imsave(folder1+'noiseRemoval.bmp',im4);

        m = im4.shape[1];
        n = im4.shape[0];
        midx = (int)(m/2);
        midy = (int)(n/2);

        r = 30
        d = 0
        dist = 0.0

        x1 = midx-r;
        x2 = midy+r;
        y1=midy-r;
        y2=midy+r;

        for i in range(1,m,1):
            for j in range(1,n,1):
                dist = (i-midx)*(i-midx) + (j-midy)*(j-midy);
                dist = math.sqrt(dist)
                d = (int)(dist+1)
                if(d <= r):
                    im4[i,j] = 0;
                    #print("hi")

                
        #skimage.io.imsave(folder1+'pupilRemoval.bmp',im4);

        im4 = np.pad(im4,(1,1),'constant',constant_values=0);
        #print(im.shape)

        im2 = skimage.io.imread('D://zeros.jpg')
        im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY);

        for i in range(1,m+1,1):
            for j in range(1,n+1,1):
                pi = im4[i,j]
                #print(pi)
                mulFactor = 1
                sumPix = 0
                
                for k in range(1,4):
                    for l in range(1,4):
                        #print(im[i-2+k,j-2+l])
                        if k==2 and l==2 :
                            continue;
                        elif im4[i-2+k,j-2+l] > pi :
                            sumPix += 1*mulFactor

                        mulFactor *= 2

                im2[i,j] = sumPix
                
        im2 = im2[1:n+1,1:m+1]
        #print(im2.shape)

        for i in range(1,m-1,1):
            for j in range(1,n-1,1):
                im2[i,j] += im4[i,j]

        #skimage.io.imsave(folder1+'localBinaryPattern.bmp',im2);
        skimage.io.imsave(folder2+file,im2);
    return;



def Iris_Feature_Extraction():
    folder1 = "D://MajorProject//IrisProcessed//";
    files = os.listdir(folder1)

    #images are of standard size 160x160
    blocks = [0,20,40,60,80,100,120,140,160]
    
    for i in range(128):
        cols.append("FEATURE_"+str(i+1));
    
    cols.append("PERSON");
    data = []
    fileCount = 0;

    for file in files :
        fileCount+=1;
        print(fileCount)
        im = skimage.io.imread(folder1+file);
        person = file[:-6];
        n = im.shape[0]
        m = im.shape[1]
        im2 = im;
        
        result = []
        result2=[]
        for k in range(1,len(blocks)):
            sum1 = 0;
            sum2 = 0;
            sum3 = 0;
            sum4 = 0;
            sum5 = 0;
            sum6 = 0;
            sum7 = 0;
            sum8 = 0;

            var1 = 0;
            var2 = 0;
            var3 = 0;
            var4 = 0;
            var5 = 0;
            var6 = 0;
            var7 = 0;
            var8 = 0;
            
            for i in range(blocks[k-1],blocks[k]):
                for j in range(0,20):
                    #print(i,j)
                    sum1 += im2[j,i]
                for j in range(20,40):
                    sum2 += im2[j,i]
                for j in range(40,60):
                    sum3 += im2[j,i]
                for j in range(60,80):
                    #print(i,j)
                    sum4 += im2[j,i]
                for j in range(80,100):
                    sum5 += im2[j,i]
                for j in range(100,120):
                    sum6 += im2[j,i]
                for j in range(120,140):
                    #print(i,j)
                    sum7 += im2[j,i]
                for j in range(140,160):
                    sum8 += im2[j,i]
                

            sum1 = int(sum1/m*n)
            sum2 = int(sum2/m*n)
            sum3 = int(sum3/m*n)
            sum4 = int(sum4/m*n)
            sum5 = int(sum5/m*n)
            sum6 = int(sum6/m*n)
            sum7 = int(sum7/m*n)
            sum8 = int(sum8/m*n)
        
            result.append(sum1)
            result.append(sum2)
            result.append(sum3)
            result.append(sum4)
            result.append(sum5)
            result.append(sum6)
            result.append(sum7)
            result.append(sum8)

            #print("result len ",len(result))
            index1 = 0
            for i in range(blocks[k-1],blocks[k]):
                for j in range(0,20):
                    if im2[j,i]>sum1 :
                        var1 += im2[j,i]-result[index1]
                    else :
                        var1 += result[index1]-im2[j,i]
                for j in range(20,40):
                    if im2[j,i]>sum2 :
                        var2 += im2[j,i]-result[index1+1]
                    else :
                        var2 += result[index1+1]-im2[j,i]
                for j in range(40,60):
                    if im2[j,i]>sum3 :
                        var3 += im2[j,i]-result[index1+2]
                    else :
                        var3 += result[index1+2]-im2[j,i]
                for j in range(60,80):
                    if im2[j,i]>sum1 :
                        var4 += im2[j,i]-result[index1+3]
                    else :
                        var4 += result[index1+3]-im2[j,i]
                for j in range(80,100):
                    if im2[j,i]>sum2 :
                        var5 += im2[j,i]-result[index1+4]
                    else :
                        var5 += result[index1+4]-im2[j,i]
                for j in range(100,120):
                    if im2[j,i]>sum3 :
                        var6 += im2[j,i]-result[index1+5]
                    else :
                        var6 += result[index1+5]-im2[j,i]
                for j in range(120,140):
                    if im2[j,i]>sum1 :
                        var7 += im2[j,i]-result[index1+6]
                    else :
                        var7 += result[index1+6]-im2[j,i]
                for j in range(140,160):
                    if im2[j,i]>sum2 :
                        var8 += im2[j,i]-result[index1+7]
                    else :
                        var8 += result[index1+7]-im2[j,i]
                

            var1 = int(var1/(m*n-1))
            var2 = int(var2/(m*n-1))
            var3 = int(var3/(m*n-1))
            var4 = int(var4/(m*n-1))
            var5 = int(var5/(m*n-1))
            var6 = int(var6/(m*n-1))
            var7 = int(var7/(m*n-1))
            var8 = int(var8/(m*n-1))

            
            result2.append(var1)
            result2.append(var2)
            result2.append(var3)
            result2.append(var4)
            result2.append(var5)
            result2.append(var6)
            result2.append(var7)
            result2.append(var8)
            
        person = 50
        for x in result2 :
            result.append(x)
        result.append(person)
        data.append(result)

    df = pd.DataFrame(data, columns = cols)
    df.to_csv("C://Python36-32//TestData4.csv");
    data = pd.read_csv("C://Python36-32//TestData4.csv")
    print(data)
    
    testDataframe = data
    return testDataframe






def IrisRecognition(testDataframe):
    data = pd.read_csv('C://Python36-32//TrainData4.csv')
    labelTrain=data[['PERSON']];
    dataTrain = data.drop(labels=['PERSON','Unnamed: 0'],axis=1)
    #print(dataTrain)
    data = pd.read_csv('C://Python36-32//TestData4.csv')
    labelTest=data[['PERSON']];
    dataTest = data.drop(labels=['PERSON','Unnamed: 0'],axis=1)
    #print(dataTest)
    model = KNeighborsClassifier(n_neighbors=1)

    model.fit(dataTrain,labelTrain)

    print("CLASSIFICATION RATE")
    print(model.score(dataTrain,labelTrain))

    print("RECOGNITION RATE")
    print(model.score(dataTest,labelTest))

    #print("RECOGNITION RATE")
    #print(model.score(dataTest,labelTest))
    recog = model.predict(dataTest)
    print("Person = ",recog)
    return recog









#-----------------------EMOTION PART-----------------------#


emotions = ["anger", "happy", "sadness"]
fishface = cv2.face.FisherFaceRecognizer_create() #Init fisher face classifier
#lbpface = cv2.face.LBPHFaceRecognizer_create() # Init LBPH classifier
data = {}




def Emotion_captureLiveImage():
    im = skimage.io.imread('D:\\MajorProject\\Face.png')
    skimage.io.imsave('D:\\MajorProject\\Emotion_Dataset\\Testing\\0.jpg',im)



def extract_Face():

    faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

    #emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    emotions = ["anger", "happy", "sadness"]
    
    files = os.listdir("D:\\MajorProject\\Emotion_Dataset\\Testing\\")
    filenumber = 0


    for f in files:
        frame = cv2.imread("D:\\MajorProject\\Emotion_Dataset\\Testing\\"+f)
        
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)


        if len(face) == 1:
            facefeatures = face
                
        elif len(face_two) == 1:
            facefeatures = face_two
                
        elif len(face_three) == 1:
            facefeatures = face_three
                    
        elif len(face_four) == 1:
            facefeatures = face_four
                    
        else:
            facefeatures = ""


        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print ("Face Detected in File : ",f)
            gray = frame[y:y+h, x:x+w] #Cut the frame to size
            
            newImg = cv2.resize(gray, (350, 350)) #Resize face so all images have same size

            cv2.imwrite("D:\\MajorProject\\Emotion_Dataset\\Testing\\"+str(filenumber)+".jpg", newImg)
            
            image = skimage.io.imread("D:\\MajorProject\\Emotion_Dataset\\Testing\\"+str(filenumber)+".jpg")
            filenumber += 1
            
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            gray = sobel(image)
        
            skimage.io.imsave("D:\\MajorProject\\Emotion_Dataset\\TestingImages\\"+str(filenumber)+".jpg", gray)

        filenumber += 1







def get_files(emotion): #split 80-20

    files = os.listdir("D:\\MajorProject\\Emotion_Dataset_sobel\\"+emotion+"\\")
    files2 = os.listdir("D:\\MajorProject\\Emotion_Dataset\\TestingImages\\")

    random.shuffle(files)
    random.shuffle(files2)

    training = files[:int(len(files)*1)]
    prediction = files2[:int(len(files2)*1)]
    #prediction = files[-int(len(files)*0.2):]

    return training, prediction






def make_sets():
    
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    
    for emotion in emotions:
        training, prediction = get_files(emotion)

        #Data appended to the training list, and generates labels 0-7
        for item in training:
            image = cv2.imread("D:\\MajorProject\\Emotion_Dataset_sobel\\"+emotion+"\\"+item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

        #Data appended to the prediction list, and generates labels 0-7
        for item in prediction:
            #print(item)
            image = cv2.imread("D:\\MajorProject\\Emotion_Dataset\\TestingImages\\"+item)
            #image = cv2.imread("D:\\MajorProject\\Emotion_Dataset\\"+emotion+"\\"+item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray);
            prediction_labels.append(emotions.index(emotion))

            #print("\n> Training_Data : ",len(training_data))
            #print("\n> Prediction_Data : ",len(prediction_data))
        
    return training_data, training_labels, prediction_data, prediction_labels








def start_recog():

    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    print("# Sets Created Sucessfully!!")
    print ("\n\n----> Training Fisher Face Classifier <----")
    #print ("> The Size of Training Set : ",len(training_labels)," Images")

    #print(training_data)
    fishface.train(training_data, np.asarray(training_labels))
    
    print ("\n> Let's Predict Classification Set")
    cnt = 0
    correct = 0
    incorrect = 0
    
    for image in prediction_data:
        pred, conf = fishface.predict(image)

        print("\n> Prediction : ",emotions[pred])
        
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1

            mixer.init()
            mixer.music.load('D:\\MajorProject\\EmotionSongs\\'+emotions[pred]+".mp3")
            mixer.music.play()
            sleep(20)
            mixer.music.stop()
                        

        else:
            incorrect += 1
            cnt += 1

    return emotions[pred]











# Main Function
print("\n--------------LIVE IRIS DETECTION AND RECOGNITION--------------");
print("\n\nLOOK AT THE CAMERA :) ");
captureLiveImage();
print("\n> Face Image Captured...");
Extract_Eyes();
print("\n> Eyes Extracted...");
IrisPreprocessing();
print("\n> Eyes Sucessfully Processed...");
testDataframe = Iris_Feature_Extraction();
print("\n> Feature Extracted...")
print("\n> Iris Trained and Recognized...");
person = IrisRecognition(testDataframe);
if person[0] == 50 :
    print("\n> Authenticated User : Determine emotion");
    auth = 1
    print("AUTH:",auth);
    
  
else:
    auth = 0
"""
    files = os.listdir("D:\\MajorProject\\Emotion_Dataset\\Testing\\")
    for file in files:
        im = skimage.io.imread("D:\\MajorProject\\Emotion_Dataset\\Testing\\"+file)
        skimage.io.imsave("D://MajorProject//Invalid User//",im);
"""


print("\n--------------EMOTION DETECTION--------------");

Emotion_captureLiveImage();
extract_Face()
emotion = start_recog()

connect(auth,emotion)

print("\n------------------------------------------------------------------\n");

