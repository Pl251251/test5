import cv2
import numpy as np
from PIL import Image
import math

#pillow
im = Image.open('test_l3.jpeg')

w,h =im.size
im2 = Image.new(im.mode, (22*w/12, 15*h/8), (255,255,255))
im2.paste(im, (int(5*w/12), int(3.5*h/8)))

im2.save("test.jpg", quality = 100)
#opencv

img = cv2.imread('test.jpg', 1)

# Locate points of the documents
# or object which you want to transform
pts1 = np.float32([[int(w/12), int(h/8)], [int(640), int(260)], [int(0), int(400)], [int(640), int(400)]])
pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])

# Apply Perspective Transform Algorithm
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, matrix, (400, 400))
# Wrap the transformed image
cv2.imshow('frame', img)  # Initial Capture
cv2.imshow('frame1', result)  # Transformed Capture

#finding a circle
blur= cv2.medianBlur(result,7)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
try:
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=246,maxRadius=265)
    info = np.uint16(np.around(circles))
    a=400

    for i in info[0,:]:
        if i[2] < a:
            a = i[2]
            b= i
    cv2.circle(result,(b[0],b[1]),b[2],(0,0,255),2)
    cv2.circle(result,(b[0],b[1]),2,(0,0,255),3)

except:
    pass


cv2.imshow('oval', result)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# Apply Perspective Transform Algorithm
matrix2 = cv2.getPerspectiveTransform(pts2, pts1)
result2 = cv2.warpPerspective(result, matrix2, (22*w/12, 15*h/8))
# Wrap the transformed image
cv2.imshow('frame', result)  # Initial Capture
cv2.imshow('frame1', result2)  # Transformed Capture


#opencv
img = cv2.imread('test.jpg',1)
dim1 = img.shape[1]
dim2 = img.shape[0]
dim3 = int((dim1/2) - w/2)+1
dim4 = int((dim2/2) -h/2)+1
dim5 = int((dim1/2) + w/2)-1
dim6 = int((dim2/2) +h/2)-1
img = img[dim4:dim6,dim3:dim5]
cv2.imwrite("final.jpg", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

cv2.imshow('oval', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
