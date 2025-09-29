# src/embed.py
import numpy as np
import joblib
from sklearn.decomposition import TruncatedSVD
from .autoencoder import train_autoencoder
import torch
import os

def has_gpu():
    """Check if CUDA GPU is available."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False

def embed_with_autoencoder(X_sparse, cfg, out_dir):
    """
    Train an autoencoder on sparse k-mer features and return embeddings.
    Saves model checkpoint and metadata.
    """
    ae_cfg = cfg['embed']['ae']
    device = ae_cfg.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if has_gpu() else 'cpu'

    model, embeddings = train_autoencoder(
        X_sparse,
        latent_dim=ae_cfg.get('latent_dim', 128),
        hidden_dims=ae_cfg.get('hidden_dims', [1024, 512]),
        batch_size=ae_cfg.get('batch_size', 1024),
        epochs=ae_cfg.get('epochs', 30),
        lr=ae_cfg.get('lr', 1e-3),
        weight_decay=ae_cfg.get('weight_decay', 1e-6),
        device=device,
        checkpoint_path=os.path.join(out_dir, "ae_best.pth"),
        early_stopping=ae_cfg.get('early_stopping_patience', 5)
    )

    # Save AE metadata
    meta_path = os.path.join(out_dir, "ae_meta.joblib")
    try:
        joblib.dump({
            'latent_dim': ae_cfg.get('latent_dim'),
            'hidden_dims': ae_cfg.get('hidden_dims')
        }, meta_path)
    except Exception as e:
        print("[embed] Failed to save AE metadata:", e)

    return embeddings, model

def embed_with_svd(X_sparse, n_components=256, out_dir=None):
    """Embed sequences with truncated SVD."""
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    emb = svd.fit_transform(X_sparse)
    if out_dir:
        svd_path = os.path.join(out_dir, "svd.joblib")
        joblib.dump(svd, svd_path)
    return emb, svd

def get_embeddings(X_sparse, cfg, out_dir):
    """
    Get sequence embeddings based on configuration.
    Supports autoencoder or SVD fallback.
    """
    method = cfg['embed'].get('method', 'autoencoder')
    if method == 'autoencoder':
        try:
            emb, _ = embed_with_autoencoder(X_sparse, cfg, out_dir)
            return emb
        except Exception as e:
            print("[embed] Autoencoder failed or not available. Falling back to SVD. Error:", e)
            return embed_with_svd(
                X_sparse,
                n_components=cfg['embed'].get('svd_dim', 256),
                out_dir=out_dir
            )[0]
    else:
        return embed_with_svd(
            X_sparse,
            n_components=cfg['embed'].get('svd_dim', 256),
            out_dir=out_dir
        )[0]
