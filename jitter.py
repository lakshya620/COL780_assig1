"""
COL-780 Assignment-1

Somanshu 2018EE10314
Lakshya  2018EE10222
"""

import cv2
import os
from utilities import laplacian_sharpening

def bg_subtraction(inp_path,model_type,eval_path,out_path):
       
    eval_frame_file = open(eval_path,'r')
    eval_frames = eval_frame_file.read()
    eval_frame_file.close()
    eval_frames = eval_frames.split()          ## reading and processing eval_frames.txt into array on integers
    for i in range(len(eval_frames)):
        eval_frames[i]= int(eval_frames[i])
    
    
    image_list = os.listdir(inp_path)
    image_list.sort()
    
    if model_type==1:
        model = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=40,detectShadows=False)
    elif model_type==2:                                               ## chosing the model
        model = cv2.createBackgroundSubtractorKNN(dist2Threshold=500,detectShadows=False)
    
    
    output_masks = []
    for i in range(len(image_list)):
        frame = cv2.imread(os.path.join(inp_path,image_list[i]))
        frame = laplacian_sharpening(frame)
        mask = model.apply(frame)

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)) 
        
                    
        if i >= (eval_frames[0]-1) and i <= (eval_frames[1]-1): 
            mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel1)
            mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel2)
            mask = cv2.GaussianBlur(mask,(5,5),0)
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
    return 


########################################################################################
"""
inp_path = "COL780_A1_Data/jitter/input"
eval_path = "COL780_A1_Data/jitter/eval_frames.txt"
out_path = "COL780_A1_Data/jitter/predicted"
mod = 2
bg_subtraction(inp_path, mod, eval_path, out_path)
"""


"""
python eval.py -p=COL780_A1_Data/jitter/predicted -g=COL780_A1_Data/jitter/groundtruth
"""