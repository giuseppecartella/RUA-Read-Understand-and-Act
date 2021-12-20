import cv2
import matplotlib.pyplot as plt

image = cv2.imread('vision_transformer/SIGNS/CLASS_1/Diapositiva6.PNG')
image = cv2.resize(image, (2,1))
plt.imshow(image)
plt.show()