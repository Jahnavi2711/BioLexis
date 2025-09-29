#!/usr/bin/env python3
"""
Configuration optimizer for the eDNA pipeline.
This script helps choose the best configuration based on your system specs.
"""

import os
import psutil
import multiprocessing
import yaml

def get_system_specs():
    """Get system specifications for optimization."""
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Check if CUDA is available
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass
    
    return {
        'cpu_count': cpu_count,
        'memory_gb': memory_gb,
        'cuda_available': cuda_available
    }

def generate_optimized_config(specs):
    """Generate an optimized configuration based on system specs."""
    
    # Determine optimal settings based on system specs
    if specs['memory_gb'] >= 32:
        # High-memory system
        n_features = 65536
        batch_size = 20000
        n_jobs = specs['cpu_count']
        num_workers = min(8, specs['cpu_count'])
    elif specs['memory_gb'] >= 16:
        # Medium-memory system
        n_features = 32768
        batch_size = 15000
        n_jobs = specs['cpu_count']
        num_workers = min(6, specs['cpu_count'])
    else:
        # Low-memory system
        n_features = 16384
        batch_size = 10000
        n_jobs = max(1, specs['cpu_count'] - 1)
        num_workers = min(4, specs['cpu_count'])
    
    # Choose embedding method based on CUDA availability
    if specs['cuda_available']:
        embed_method = "autoencoder"
        ae_epochs = 20
    else:
        embed_method = "svd"
        ae_epochs = 10
    
    config = {
        'seed': 42,
        'io': {
            'reference_csv': "data/raw/reference.csv",
            'labels_csv': "data/raw/input.fasta",
            'model_dir': "results/models"
        },
        'map_dir': "results/reference_map",
        'preprocess': {
            'seq_col': "sequence",
            'taxonomy_col': "taxonomy",
            'organism_col': "organism_name",
            'max_n_frac': 0.05,
            'min_seq_len': 50
        },
        'kmers': {
            'k': [4, 6],  # Reduced for speed
            'n_features': n_features,
            'normalize': True,
            'batch_size': batch_size,
            'n_jobs': n_jobs
        },
        'embed': {
            'method': embed_method,
            'svd_dim': 128,
            'ae': {
                'latent_dim': 64,
                'hidden_dims': [512, 256],
                'batch_size': 2048,
                'epochs': ae_epochs,
                'lr': 2e-3,
                'weight_decay': 1e-6,
                'device': 'auto',
                'checkpoint_every': 5,
                'early_stopping_patience': 3,
                'num_workers': num_workers
            }
        },
        'cluster': {
            'umap_n_neighbors': 30,
            'umap_min_dist': 0.1,
            'umap_2d_components': 2,
            'umap_10d_components': 8,
            'hdbscan_min_cluster_size': 50,
            'hdbscan_min_samples': 5,
            'emergent_taxa_percentile': 95
        },
        'label_transfer': {
            'ranks': ["level_1","level_2","level_3","level_4","level_5","level_6","level_7","level_8"],
            'min_confidence': 0.6
        },
        'decision_engine': {
            'min_classification_confidence': 0.85,
            'known_islands_library_path': "data/known_islands_embeddings.npy",
            'known_islands_distance_threshold': 0.1
        },
        'abundance': {
            'group_by': "assignment"
        },
        'outputs': {
            'parquet_compression': "snappy",
            'report_html': "report.html"
        }
    }
    
    return config

def main():
    """Generate optimized configuration."""
    print("🔍 Analyzing your system...")
    specs = get_system_specs()
    
    print(f"💻 System specs:")
    print(f"   CPU cores: {specs['cpu_count']}")
    print(f"   Memory: {specs['memory_gb']:.1f} GB")
    print(f"   CUDA available: {specs['cuda_available']}")
    
    print("\n⚡ Generating optimized configuration...")
    config = generate_optimized_config(specs)
    
    # Save the optimized configuration
    with open('configs/optimized.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("✅ Optimized configuration saved to 'configs/optimized.yaml'")
    print("\n🚀 To use the optimized configuration, run:")
    print("   python -m src.pipeline --config configs/optimized.yaml --input data/raw/input.fasta --reference data/raw/labels.csv --out results/run1")

if __name__ == "__main__":
    main()
