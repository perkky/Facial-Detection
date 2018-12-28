import numpy
import cv2

faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayFrame, scaleFactor=1.2, minNeighbors=5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),((x+w),(y+h)), (0,255,0), 3)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break;

cap.release()
cv2.destroyAllWindows()