import cv2
import numpy as np

frame = 255 * np.ones((300,300,3), dtype=np.uint8)

cv2.imwrite("out.jpg", frame)
print("saved out.jpg")import cv2

frame = 255 * np.ones((300,300,3), dtype=np.uint8)
cv2.imwrite("frame.jpg", frame)
print("saved frame")

