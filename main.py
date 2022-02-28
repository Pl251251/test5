import cv2
import numpy as np


img = cv2.imread('test_l3.jpeg', 1)
width = img.shape[1]
height = img.shape[0]
print(width)
print(height)
# Locate points of the documents
# or object which you want to transform
pts1 = np.float32([[0, 260], [640, 260], [0, 400], [640, 400]])
pts2 = np.float32([[0, 0], [400, 0], [0, 640], [400, 640]])

# Apply Perspective Transform Algorithm
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, matrix, (500, 600))
# Wrap the transformed image
cv2.imshow('frame', img)  # Initial Capture
cv2.imshow('frame1', result)  # Transformed Capture


cv2.imwrite("test.jpg", result, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.waitKey(0)
cv2.destroyAllWindows()
