"""
COL-780 Assignment-1

Somanshu 2018EE10314
Lakshya  2018EE10222
"""

import cv2
import os
import numpy as np

def show_img(img):
    cv2.imshow("Mask",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def laplacian_sharpening(image):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    image = cv2.filter2D(image, -1, kernel)
    return image

class Simple_avg:
    def __init__(self,eval_frames,inp_path):
        self.start=eval_frames[0]
        self.inp_path=inp_path
        img_names = os.listdir(inp_path)
        frame_list=[]
        for i in range(0,self.start):
            frame = cv2.imread(os.path.join(self.inp_path,img_names[i]))
            frame_list.append(frame)             
        frame_list=np.array(frame_list)
        avg_frame=np.mean(frame_list,axis=0)
        self.avg_frame=avg_frame.astype(np.uint8)
        self.avg_frame=cv2.cvtColor(self.avg_frame, cv2.COLOR_BGR2GRAY)
        
        
    def apply(self,img):
        mask=cv2.subtract(self.avg_frame,cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
        (thresh, blackAndWhiteImage) = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
        return blackAndWhiteImage
        
    
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
        model = cv2.createBackgroundSubtractorKNN(dist2Threshold=500,detectShadows=False)
    elif model_type==3:
        model=Simple_avg(eval_frames,inp_path)

    output_masks = []
    for i in range(len(image_list)):
        frame = cv2.imread(os.path.join(inp_path,image_list[i]))
        frame = cv2.bilateralFilter(frame,5,50,25)
        mask = model.apply(frame)                         

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
       
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
    return 



#################################################################################
inp_path = "COL780_A1_Data/moving_bg/input"
eval_path = "COL780_A1_Data/moving_bg/eval_frames.txt"
out_path = "COL780_A1_Data/moving_bg/predicted"
mod = 2
bg_subtraction(inp_path, mod, eval_path, out_path)

"""
python eval.py -p=COL780_A1_Data/moving_bg/predicted -g=COL780_A1_Data/moving_bg/groundtruth
"""
