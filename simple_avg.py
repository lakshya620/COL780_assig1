"""
COL-780 Assignment-1

Somanshu 2018EE10314
Lakshya  2018EE10222
"""
import cv2
import os
import numpy as np

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