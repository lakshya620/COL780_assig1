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
       
        # print((self.avg_frame))
        
        
    def apply(self,img):
        mask=cv2.subtract(self.avg_frame,cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
        # mask=img-self.avg_frame
        (thresh, blackAndWhiteImage) = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)

        # cv2.normalize(mask,mask,0,255,cv2.NORM_MINMAX)
        return blackAndWhiteImage
    
class Running_avg:
    def __init__(self,inp_path,eval_frames,count=100):
        self.inp_path=inp_path
        self.count=count
        self.start=0
        self.frame_list=[]
        self.filled=False
       
    def apply(self,img):
        start=self.start
        img_names = os.listdir(inp_path)
        if not self.filled:
            for i in range(max(start-self.count,0),start+1):
                frame = cv2.imread(os.path.join(self.inp_path,img_names[i]))
                self.frame_list.append(frame)
        else:
            self.frame_list.append(img)
            self.frame_list.pop(0)
        if(len(self.frame_list)==self.count):
            self.filled=True
        frame_list=np.array(self.frame_list)
        avg_frame=np.mean(frame_list,axis=0)
        avg_frame=avg_frame.astype(np.uint8)
        avg_frame=cv2.cvtColor(avg_frame, cv2.COLOR_BGR2GRAY)
        mask=cv2.subtract(avg_frame,cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
        (thresh, blackAndWhiteImage) = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
        self.start+=1
        
        return blackAndWhiteImage       
        
    
    
    
def bg_subtraction(inp_path,model_type,eval_path,out_path):
       
    eval_frame_file = open(eval_path,'r')
    eval_frames = eval_frame_file.read()
    eval_frame_file.close()
    eval_frames = eval_frames.split()          ## reading and processing eval_frames.txt into array on integers
    for i in range(len(eval_frames)):
        eval_frames[i]= int(eval_frames[i])
    print(eval_frames)    
    
    image_list = os.listdir(inp_path)
    
    if model_type==1:
        model = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=40,detectShadows=False)
    elif model_type==2:                                               ## chosing the model
        model = cv2.createBackgroundSubtractorKNN(dist2Threshold=500,detectShadows=False)
    elif model_type==3:
        model=Simple_avg(eval_frames,inp_path)
    else:
        model=Running_avg(inp_path,eval_frames)
    # show_img(model.avg_frame)
    output_masks = []
    for i in range(len(image_list)):
        frame = cv2.imread(os.path.join(inp_path,image_list[i]))
        # frame = cv2.bilateralFilter(frame,5,50,25)
        frame=cv2.GaussianBlur(frame,(5,5),0)
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



inp_path = "COL780_A1_Data/moving_bg/input"
eval_path = "COL780_A1_Data/moving_bg/eval_frames.txt"
out_path = "COL780_A1_Data/moving_bg/predicted"
mod = 4
# model=Simple_avg([10,20], inp_path)
bg_subtraction(inp_path, mod, eval_path, out_path)
