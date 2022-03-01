import cv2
import numpy as np
from PIL import Image
#import math

#pillow
im = Image.open('test_l3.jpg')

w,h =im.size
print(w)
print(h)
im2 = Image.new(im.mode, (int(22*w/12), int(15*h/8)), (255,255,255))
im2.paste(im, (int(5*w/12), int(3.5*h/8)))
print(im2.size)


im2.save("test.jpg", quality = 100)
#opencv

img = cv2.imread('test.jpg', 1)


# Locate points of the documents
# or object which you want to transform
pts1 = np.float32([[0, int(5.4*h/8)], [int(14.25*w/12), 0], [int(7.75*w/12), int(15*h/8)], [int(22*w/12), int(6.7*h/8)]])
pts2 = np.float32([[0, 0], [800, 0], [0, 800], [800, 800]])

# Apply Perspective Transform Algorithm
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, matrix, (800, 1000))
# Wrap the transformed image

#cv2.imshow('frame1', result)  # Transformed Capture
cv2.imwrite("test2.jpg", result)

#finding a circle
blur= cv2.medianBlur(result,7)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
try:
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=110,maxRadius=130)
    info = np.uint16(np.around(circles))
    a=400
    """
    for i in info[0,:]:
        if i[2] < a:
            a = i[2]
            b= i
    cv2.circle(result,(b[0],b[1]),b[2],(0,0,255),2)
    cv2.circle(result,(b[0],b[1]),2,(0,0,255),3)
    """
    for i in info[0, :]:
        cv2.circle(result, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)
except:
    pass


cv2.imshow('oval', result)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# Apply Perspective Transform Algorithm
matrix2 = cv2.getPerspectiveTransform(pts2, pts1)
result2 = cv2.warpPerspective(result, matrix2, (int(22*w/12), int(16*h/8)))
# Wrap the transformed image

cv2.imshow('frame1', result2)  # Transformed Capture



dim1 = result2.shape[1]
dim2 = result2.shape[0]
dim3 = int((dim1/2) - w/2)+1
dim4 = int((dim2/2) -h/2)-35
dim5 = int((dim1/2) + w/2)-1
dim6 = int((dim2/2) +h/2)-35
img = result2[dim4:dim6,dim3:dim5]
cv2.imwrite("final.jpg", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


cv2.waitKey(5000)
cv2.imshow('oval', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
