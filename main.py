import cv2
import time

#On my laptop
#0 front camera
#1 back camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

for x in range(10):
    cv2.imshow('frame', frame)
    out = cv2.imwrite('./capture' + str(x) + '.jpg', frame)
    cv2.waitKey(1)
    time.sleep(3)
    

cap.release()