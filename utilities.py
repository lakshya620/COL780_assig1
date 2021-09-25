"""
COL-780 Assignment-1

Somanshu 2018EE10314
Lakshya  2018EE10222
"""
import cv2
import numpy as np
           
def show_img(img):
    cv2.imshow("Mask",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def laplacian_sharpening(image):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    image = cv2.filter2D(image, -1, kernel)
    return image

def shadow_removal(img):
    rgb_channels= cv2.split(img)
    result_channels=[None]*3
    kernel=np.ones((7,7),dtype=np.uint8)
    
    for i in range(3):
        channel = rgb_channels[i]
        img = cv2.dilate(channel, kernel)
        img = cv2.medianBlur(img, 5)
        img = 255 - cv2.absdiff(channel, img)
        result_channels[i] = img
    
    img = cv2.merge(result_channels)
    return img