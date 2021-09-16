# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 17:28:02 2021

@author: lenovo
"""
import numpy as np
import os
from openCV_models import bg_subtraction
def getScore(resfile,command):
    os.system(command)
    with open(resfile,'r') as f:
        line =f.readline()
        lis=(line.split(':'))
        score=lis[-1]
        return float(score)
    
def hyperparam_tune(predictor,inp_path, mog, eval_path, out_path,eval_command,params_range,resfile,seed=42,num_iter=100):
    '''
    predictor :function to be used
    params_range:dict containing all lists for different params
    textfile for writing params used 
    ''' 
# =============================================================================
#     get params and genearate params_dict
#     feed into method generating predictions
#     run method 
#     append result into text file along with params used 
#     pick out best parameter
# =============================================================================
    dia_list=params_range['pre_filter_dia']
    sigmaSpace_list= params_range['pre_filter_sigmaSpace']
    sigmaColor_list=params_range['pre_filter_sigmaColor']
    k1_size_list=params_range['k1']
    k2_size_list=params_range['k2']
    maxScore=0
    best_params={}
    # for knn 
    dist2Threshold_list=params_range['dist2Threshold']
    np.random.seed(seed)
    for i in range(num_iter):
        dia=np.random.choice(dia_list)
        sigmaSpace= np.random.choice(sigmaSpace_list)
        sigmaColor=np.random.choice(sigmaColor_list)
        k1_size=np.random.choice(k1_size_list)
        k2_size=np.random.choice(k2_size_list)
        
        dist2Threshold=np.random.choice(dist2Threshold_list)
        params={"dist2Threshold":dist2Threshold,'varThreshold':500,
                'history':40,'pre_filter_dia':dia,
        'pre_filter_sigmaSpace':sigmaSpace,
        'pre_filter_sigmaColor':sigmaColor,
        'k1':k1_size,
        'k2':k2_size,
        'mog_used':False
        }
        predictor(inp_path, mog, eval_path, out_path,params)
        os.system(eval_command)
        temp_score=getScore(resfile, eval_command)
        if(temp_score > maxScore):
            maxScore=temp_score
            best_params=params
    
    print(maxScore)
    print(best_params)       
       
        

if __name__=='__main__':
    inp_path = "COL780_A1_Data/baseline/input"
    eval_path = "COL780_A1_Data/baseline/eval_frames.txt"
    out_path = "COL780_A1_Data/baseline/predicted"
    mog = False
    eval_command='python eval.py -p=COL780_A1_Data/baseline/predicted -g=COL780_A1_Data/baseline/groundtruth > res.txt'
    dia_list=np.arange(1, 20,1)
    sigmaSpace_list= np.arange(5,100,5)
    sigmaColor_list= np.arange(5,100,5)
    k1_size_list=np.arange(3,20,2)
    k2_size_list=np.arange(3,20,2)
    
    # for knn 
    dist2Threshold_list=np.arange(100,1000,100)
    
    params_range={}
    params_range['pre_filter_dia']=dia_list
    params_range['pre_filter_sigmaSpace']=sigmaSpace_list
    params_range['pre_filter_sigmaColor']=sigmaColor_list
    params_range['k1']=k1_size_list
    params_range['k2']=k2_size_list
    params_range['dist2Threshold']=dist2Threshold_list
    
    hyperparam_tune(bg_subtraction, inp_path, mog, eval_path, out_path, eval_command, params_range,'res.txt')
    