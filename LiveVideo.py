import numpy as np
import cv2
import random

cap = cv2.VideoCapture(0)

i=0
count = 0

while(count!=1):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        i+=1
        cv2.imwrite('D://Char_Recog//Actual_Char//{index}.png'.format(index=i),frame)
        count+=1
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
