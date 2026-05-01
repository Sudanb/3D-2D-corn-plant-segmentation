"""
train_efficientnet.py
=====================
Train EfficientNet-B3 regression model.
Usage: python train_efficientnet.py
"""

from reg_model    import EfficientNetReg
from train_engine import run_training

cfg = {
    "run_name"     : "efficientnet_b3",
    "epochs"       : 20,
    "batch_size"   : 16,
    "lr"           : 1e-4,    # encoder LR (pretrained — keep low)
    "lr_head"      : 1e-3,    # head LR
    "weight_decay" : 1e-4,
    "patience"     : 15,
    "num_workers"  : 0,
}

if __name__ == "__main__":
    model = EfficientNetReg(freeze_encoder=False, dropout=0.3)
    run_training(cfg, model)
