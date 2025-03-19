import numpy as np 
import cv2 as cv 

img = cv.imread('Resources/Photos/cat.jpg')
cv.imshow('Cat',img)

blank = np.zeros((700,500,3),dtype='uint8')
blank[:] = 0,0,0
# blank[200:300]

# blank[200:300,300:400] = 255,0,255
cv.rectangle(blank,(0,0),(250,500),(0,250,0),thickness = cv.FILLED)
cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),40,(0,0,233))

cv.line(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(255,255,255),thickness=4)
cv.putText(blank,"Hello my name is Ayush",(0,255),cv.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),2)

cv.imshow('Text',blank)
cv.waitKey(0)









