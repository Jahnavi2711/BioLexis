# src/diversity.py
import numpy as np

def shannon(p):
    p = np.array(p)
    p = p[p>0]
    return -np.sum(p * np.log(p))

def simpson(p):
    p = np.array(p)
    return 1.0 - np.sum(p**2)

def compute_alpha(abundance_vector):
    p = np.array(abundance_vector, dtype=float)
    s = p.sum()
    if s == 0:
        return {'richness': 0, 'shannon': 0.0, 'simpson': 0.0}
    p = p / s
    return {'richness': (p>0).sum(), 'shannon': float(shannon(p)), 'simpson': float(simpson(p))}
