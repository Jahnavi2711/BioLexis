# src/io.py
import os
import yaml
from pathlib import Path
import joblib
import random
import numpy as np

def read_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def ensure_dirs(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def save_joblib(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_joblib(path):
    return joblib.load(path)
