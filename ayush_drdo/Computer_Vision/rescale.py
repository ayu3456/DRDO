
import cv2 as cv

def rescaleFrame(frame,scale = 0.75):

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[1] * scale) 

    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)  





# capture = cv.VideoCapture('Resources/Videos/dog.mp4') 

# while True:

#     isTrue , frame = capture.read()
#     scaled_Frame = rescaleFrame(frame,0.25)
#     cv.imshow('Video',scaled_Frame)

#     if cv.waitKey(20) and 0xFF == ord('d'):
#         break 

# capture.release()
# cv.destroyAllWindows()


img = cv.imread('Resources/Photos/cat.jpg')
resized_img = rescaleFrame(img,0.2)
cv.imshow('Cat', resized_img)
cv.waitKey(0)





