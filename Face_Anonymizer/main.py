import cv2
import mediapipe as mp

#read image
img_path = r'C:\Users\Jinesh\Pictures\image.png'
img = cv2.imread(img_path)




cv2.imshow('img',img)
cv2.waitKey(5000)
cv2.destroyAllWindows()