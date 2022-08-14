import cv2
import numpy as np
image = cv2.imread('a.jpg', cv2.IMREAD_UNCHANGED)
image=cv2.resize(image,(100,200))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

contours = cv2.findContours(gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
print(contours)
for cnt in contours:
    (x,y,w,h) = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255))
cv2.imshow("hello",image)
cv2.waitKey(0)
cv2.destroyAllWindows()