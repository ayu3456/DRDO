import cv2 as cv 

img = cv.imread('Resources/Photos/park.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# cv.imshow("Boston",gray)


# how to blur an image 

blur = cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT)

cv.imshow('Blur',blur)


# how to find an edges in an img 
# we will use canny to find edges in an image

canny = cv.Canny(blur,125,175) 
cv.imshow("Canny Edges",canny) 



# how to dilate the image 

dilated = cv.dilate(canny,(7,7),iterations=3)
cv.imshow('Dilated',dilated)

# how to erode an image 
# it means that get back to prev state. 

eroded = cv.erode(dilated,(7,7),iterations=3)
cv.imshow("Eroded",eroded) 

# how to resize an image 
resized = cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)

cv.imshow("resized",resized)



# how to crop an image 
print(img)
cropped = img[50:200,200:400]
cv.imshow("Cropped",cropped)
cv.waitKey(0) 















