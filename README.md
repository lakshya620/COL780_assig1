# Background Subtraction    
We have tried three models for this problem:     
1. Mixture of Gaussians
2. K Nearest Neighbours


## Running Mode

Generate the prediction masks

```bash
python main.py  --inp_path=<path to input frames> --out_path=<path to generated masks> --eval_frames=<path to eval_frames.txt file> --category="<b/i/j/m/p>"
```

Evaluate the prediction masks against ground truth

```bash
python eval.py  --pred_path=<path to generated masks folder> --gt_path=<path to groundtruth masks folder>
    
