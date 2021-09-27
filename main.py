"""
COL-780 Assignment-1

Somanshu 2018EE10314
Lakshya  2018EE10222
"""

import os
import argparse
from baseline import bg_subtraction as b
from illumination import bg_subtraction as i
from jitter import bg_subtraction as j
from moving_bg import bg_subtraction as m
from ptz import bg_subtraction as p



def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='input', required=True, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='result', required=True, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=True, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m/p)")
    parser.add_argument('-e', '--eval_frames', type=str, default='eval_frames.txt', required=True, \
                                                        help="Path to the eval_frames.txt file")
    args = parser.parse_args()
    return args


def baseline_bgs(args):
    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)
    b(args.inp_path,2,args.eval_frames,args.out_path)
    return


def illumination_bgs(args):
    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)
    i(args.inp_path,1,args.eval_frames,args.out_path)
    return


def jitter_bgs(args):
    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)
    j(args.inp_path,2,args.eval_frames,args.out_path)
    return


def dynamic_bgs(args):
    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)
    m(args.inp_path,2,args.eval_frames,args.out_path)
    return


def ptz_bgs(args):
    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)
    p(args.inp_path,1,args.eval_frames,args.out_path)
    return


def main(args):
    if args.category not in "bijmp":
        raise ValueError("category should be one of b/i/j/m/p - Found: %s"%args.category)
    FUNCTION_MAPPER = {
            "b": baseline_bgs,
            "i": illumination_bgs,
            "j": jitter_bgs,
            "m": dynamic_bgs,
            "p": ptz_bgs
        }

    FUNCTION_MAPPER[args.category](args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
    
"""
python main.py -i=COL780_A1_Data/baseline/input -o=COL780_A1_Data/baseline/predicted -e=COL780_A1_Data/baseline/eval_frames.txt -c="b"
"""