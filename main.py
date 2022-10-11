import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

faceCascade = cv.CascadeClassifier('C:\\Users\\Valdemaras\\Desktop\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

face_id = input('enter user id: ')
print("Look the camera and wait ...")
count = 0

while True:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(20,20))
    
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count += 1
    
    
        cv.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])   
        cv.imshow('video',img)
    
            
    k = cv.waitKey(100) & 0xff
    if k == 27:
        break
    
    elif count >= 30: # Take 30 face sample and stop video
         break    
    
print("Face sample done")  
cap.release()
cv.destroyAllWindows()