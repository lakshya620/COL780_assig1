"""
COL-780 Assignment-1

Somanshu 2018EE10314
Lakshya  2018EE10222
"""

import cv2
import os
import numpy as np

    
def laplacian_sharpening(image):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    image = cv2.filter2D(image, -1, kernel)
    return image

def bg_subtraction(inp_path,mog,eval_path,out_path):
       
    eval_frame_file = open(eval_path,'r')
    eval_frames = eval_frame_file.read()
    eval_frame_file.close()
    eval_frames = eval_frames.split()          ## reading and processing eval_frames.txt into array on integers
    for i in range(len(eval_frames)):
        eval_frames[i]= int(eval_frames[i])
        
    
    image_list = os.listdir(inp_path)
    
    if mog:
        model = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=40,detectShadows=False)
    else:                                               ## chosing the model
        model = cv2.createBackgroundSubtractorKNN(dist2Threshold=500,detectShadows=False)
        
    output_masks = []
    for i in range(len(image_list)):
        frame = cv2.imread(os.path.join(inp_path,image_list[i]))
        frame = cv2.bilateralFilter(frame,5,50,25) 
        mask = model.apply(frame)                         ## updating the backgorund model and storing the masks obtained
        
        if not mog:
            kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)) 
        else:
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)) 
                    
        if i >= (eval_frames[0]-1) and i <= (eval_frames[1]-1):    
            mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel1)
            mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel2)
            mask = laplacian_sharpening(mask)
            output_masks.append(mask)
     
    start = eval_frames[0]        
    for mask in output_masks:
        name = str(start) +".png"
        prefix = "gt"
        num_zeros = 6 - (len(name)-4)
        for  i in range(num_zeros):
            prefix += "0"                   ## saving the predictions into png format
        name = prefix+name
        
        cv2.imwrite(os.path.join(out_path,name),mask)
        start += 1      
        
    cv2.destroyAllWindows()

    return 



inp_path = "COL780_A1_Data/baseline/input"
eval_path = "COL780_A1_Data/baseline/eval_frames.txt"
out_path = "COL780_A1_Data/baseline/predicted"
mog = False
bg_subtraction(inp_path, mog, eval_path, out_path)

"""
python eval.py -p=COL780_A1_Data/baseline/predicted -g=COL780_A1_Data/baseline/groundtruth
"""