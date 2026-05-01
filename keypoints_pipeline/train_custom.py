"""
train_custom.py
===============
Train custom CNN regression model from scratch.
Usage: python train_custom.py
"""

from reg_model    import CustomCNNReg
from train_engine import run_training

cfg = {
    "run_name"     : "custom_cnn",
    "epochs"       : 80,      # longer — no pretrained features
    "batch_size"   : 16,
    "lr"           : 3e-4,    # single LR (no pretrained encoder)
    "lr_head"      : 3e-4,
    "weight_decay" : 1e-4,
    "patience"     : 20,      # more patience — slower convergence
    "num_workers"  : 0,
}

if __name__ == "__main__":
    model = CustomCNNReg(dropout=0.4)
    run_training(cfg, model)
