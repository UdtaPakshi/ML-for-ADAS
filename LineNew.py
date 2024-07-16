import numpy as np
import cv2

img = cv2.imread("")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 75, 150)
cv2.imshow("Image", img)
cv2.imshow("Edges", edges)
cv2.waitkey(0)
cv2.destroyAllWindows()