"""
COL-780 Assignment-1

Somanshu 2018EE10314
Lakshya  2018EE10222
"""

import cv2
import os
import numpy as np


def laplacian_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    return image


def bg_subtraction(inp_path, mog, eval_path, out_path,params):
# add params
    eval_frame_file = open(eval_path, 'r')
    eval_frames = eval_frame_file.read()
    eval_frame_file.close()
    # reading and processing eval_frames.txt into array on integers
    eval_frames = eval_frames.split()
    for i in range(len(eval_frames)):
        eval_frames[i] = int(eval_frames[i])

    image_list = os.listdir(inp_path)

    if mog:
        varThreshold=params['varThreshold']
        history=params['history']
        
        model = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=40, detectShadows=False)
    else:  # chosing the model
        dist2Threshold=params['dist2Threshold']
        model = cv2.createBackgroundSubtractorKNN(
            dist2Threshold=dist2Threshold, detectShadows=False)

    output_masks = []
    dia=params['pre_filter_dia']
    sigmaSpace= params['pre_filter_sigmaSpace']
    sigmaColor=params['pre_filter_sigmaColor']
    k1_size=params['k1']
    k2_size=params['k2']
          
    for i in range(len(image_list)):
        # image preprocessing usingbilateral filter
        
        frame = cv2.imread(os.path.join(inp_path, image_list[i]))
        frame = cv2.bilateralFilter(frame, dia, sigmaColor, sigmaSpace)
        # updating the backgorund model and storing the masks obtained
        mask = model.apply(frame)
        
        if not mog:
            kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (k1_size, k1_size))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (k2_size, k2_size))
        else:
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1_size, k1_size))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2_size, k2_size))

        if i >= (eval_frames[0]-1) and i <= (eval_frames[1]-1):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
            mask = laplacian_sharpening(mask)
            output_masks.append(mask)

    start = eval_frames[0]
    for mask in output_masks:
        name = str(start) + ".png"
        prefix = "gt"
        num_zeros = 6 - (len(name)-4)
        for i in range(num_zeros):
            prefix += "0"  # saving the predictions into png format
        name = prefix+name

        cv2.imwrite(os.path.join(out_path, name), mask)
        start += 1

    return
if __name__=='__main__':
    inp_path = "COL780_A1_Data/baseline/input"
    eval_path = "COL780_A1_Data/baseline/eval_frames.txt"
    out_path = "COL780_A1_Data/baseline/predicted"
    mog = False
    params={"dist2Threshold":500,'varThreshold':500,
        'history':40,'pre_filter_dia':5,
    'pre_filter_sigmaSpace':50,
    'pre_filter_sigmaColor':25,
    'k1':3,
    'k2':7,
    'mog_used':False
    }
    # command='python eval.py -p=COL780_A1_Data/baseline/predicted -g=COL780_A1_Data/baseline/groundtruth')
    bg_subtraction(inp_path, mog, eval_path, out_path,params)

"""
python eval.py -p=COL780_A1_Data/baseline/predicted -g=COL780_A1_Data/baseline/groundtruth
"""
