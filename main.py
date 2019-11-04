import imutils
import numpy as np
import time
import cv2
from imutils.object_detection import non_max_suppression

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap=cv2.VideoCapture('modified video 1.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#print(total)
frame_no=0
while (cap.isOpened):
    ret, frame = cap.read()
    if (ret):
        frame = imutils.resize(frame, width=500)
        frame_no=frame_no+1
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (boxes, weights) = hog.detectMultiScale(frame, winStride=(4, 4),padding=(8, 8), scale=1.05)
        #boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        boxes = non_max_suppression(boxes, probs=None, overlapThresh=0.65)
        
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = frame[y:y + h, x:x + w]
            
            #print(frame_no)
            if (frame_no==total-20):
                cv2.imwrite('person.jpg', roi_color)

        cv2.imshow("Frame",frame)
        cv2.waitKey(1)


        if (frame_no==total-20):
            break
        #cv2.imshow("Frame",roi_color)
cap.release()
cv2.destroyAllWindows()
cap1=cv2.VideoCapture('modified video 2.mp4')
frame_no=0
while (cap1.isOpened):
    ret, frame = cap1.read()
    if (ret):
        frame = imutils.resize(frame, width=500)
        frame_no=frame_no+1
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (boxes, weights) = hog.detectMultiScale(frame, winStride=(4, 4),padding=(1, 1), scale=1.05)
        #boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        boxes = non_max_suppression(boxes, probs=None, overlapThresh=0.65)
        
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = frame[y:y + h, x:x + w]
            
            #print(frame_no)
            if (frame_no==50):
                cv2.imwrite('person_out.jpg', roi_color)

        #cv2.imshow("Frame",frame)
        cv2.imshow("Frame",frame)
        cv2.waitKey(1)

        if (frame_no==total-20):
            break
        #cv2.imshow("Frame",roi_color)
cap1.release()

image1=cv2.imread('person.jpg')
image2=cv2.imread('person_out.jpg')
hist1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
hist1 = cv2.normalize(hist1, hist1).flatten()
hist2 = cv2.calcHist([image2], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
hist2 = cv2.normalize(hist2, hist2).flatten()
result=cv2.compareHist(hist1,hist2,cv2.HISTCMP_BHATTACHARYYA)
if ((result*100)>60):
    s="Similarity, ",result*100
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 50
    fontColor              = (255,255,255)
    lineType               = 2  
    print(s)
    cv2.imshow("In first camera", image1)
    cv2.moveWindow("In first camera",200,200)
    image2=cv2.putText(image2,'yes', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
    #image2=cv2.putText(image2,s,(00, 185),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2,cv2.LINE_AA,False)
    cv2.imshow("In second camera", image2)
    cv2.waitKey(0)
cv2.destroyAllWindows()


    


