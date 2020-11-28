import cv
import time
import numpy as np

cv_car=cv.CascadeClassifier(r'C:\Users\DEBIPRASAD\Desktop\Projetc Work\ComputerVision-Projects-master\CarPedestrianDetection\cascades\haarcascade_car.xml')
capture=cv.VideoCapture(r'C:\Users\DEBIPRASAD\Desktop\Projetc Work\ComputerVision-Projects-master\CarPedestrianDetection\files\cars.avi')

while capture.isOpened():
    response,frame=capture.read()
    if response:
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        cars=cv_car.detectMultiScale(gray,1.2,3)
        for (x,y,w,h) in cars:
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),3)
            cv.imshow('cars',frame)
        if cv.waitkey(1) & 0xFF==ord('q'):
            break
    else:
        break
capture.release()
cv.destroyAllWindows()        