"""
COL-780 Assignment-1

Somanshu 2018EE10314
Lakshya  2018EE10222
"""

import cv2
import os
import numpy as np
from simple_avg import Simple_avg

def show_img(img):
    cv2.imshow("Mask",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def unsharp_masking(frame):
    image=cv2.GaussianBlur(frame, (5, 5), 0)
    image =cv2.addWeighted(frame, 1.5, image, -0.5,0.0)
    return image
def laplacian_sharpening(image):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    image = cv2.filter2D(image, -1, kernel)
    return image
        
def shadow_removal(img):
    rgb_channels= cv2.split(img)

    result_channels = []

    kernel=np.ones((7,7), np.uint8)
    for channel in rgb_channels:
        img = cv2.dilate(channel, kernel)
        img = cv2.medianBlur(img, 21)
        img = 255 - cv2.absdiff(channel, img)
        result_channels.append(img)
    
    ans = cv2.merge(result_channels)
    return ans
    
def bg_subtraction(inp_path,model_type,eval_path,out_path):
       
    eval_frame_file = open(eval_path,'r')
    eval_frames = eval_frame_file.read()
    eval_frame_file.close()
    eval_frames = eval_frames.split()          ## reading and processing eval_frames.txt into array on integers
    for i in range(len(eval_frames)):
        eval_frames[i]= int(eval_frames[i])
    
    
    image_list = os.listdir(inp_path)
    
    if model_type==1:
        model = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=40,detectShadows=False)
    elif model_type==2:                                               ## chosing the model
        model = cv2.createBackgroundSubtractorKNN(dist2Threshold=800,detectShadows=False)
    elif model_type==3:
        model=Simple_avg(eval_frames,inp_path)

    output_masks = []
    for i in range(len(image_list)):
        frame = cv2.imread(os.path.join(inp_path,image_list[i]))
        frame = cv2.bilateralFilter(frame,5,200,25)
        
        
        frame=shadow_removal(frame)                                  
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
        l,a,b=cv2.split(frame)
        
        l= cv2.adaptiveThreshold(l, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 199, 5)
        
        frame=cv2.merge((l,a,b));
        frame=cv2.cvtColor(frame,cv2.COLOR_LAB2BGR);
        name=f"{i}.png"
        cv2.imwrite(os.path.join(out_path+"/preprocessed/",name),frame)
        mask = model.apply(frame)                         

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
       
        if i >= (eval_frames[0]-1) and i <= (eval_frames[1]-1):    
            mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel1)
            mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel2)
            mask = laplacian_sharpening(mask)
            mask=cv2.resize(mask,(320,240))
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



#################################################################################
inp_path = "COL780_A1_Data/illumination/input"
eval_path = "COL780_A1_Data/illumination/eval_frames.txt"
out_path = "COL780_A1_Data/illumination/predicted"
mod = 2
bg_subtraction(inp_path, mod, eval_path, out_path)

"""
python eval.py -p=COL780_A1_Data/illumination/predicted -g=COL780_A1_Data/illumination/groundtruth
"""