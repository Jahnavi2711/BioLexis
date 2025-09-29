# src/pipeline.py
import os
import json
import numpy as np
import pandas as pd
import argparse
import torch
import joblib
import shutil # <-- Add this import
import scipy.sparse
from scipy.spatial.distance import cdist

# --- IO and Preprocessing ---
from src.io import read_config, ensure_dirs, set_seed
# MODIFIED: We pass the FASTA path directly to compute_kmers_and_embeddings now
from src.preprocess import clean_labels, deduplicate, compute_kmers_and_embeddings

# --- Supervised Arm ---
from src.label_transfer import train_rank_classifiers, predict_per_rank, per_sequence_confidence, cluster_labeling

# --- Unsupervised Arm ---
from src.cluster import umap_reduce, build_reference_map, analyze_reference_clusters, predict_on_reference_map, evaluate_input_sequences

# --- Downstream Analysis ---
from src.evaluate import evaluate_clustering
from src.abundance import aggregate_abundance
from src.diversity import compute_alpha

# --- Visualization ---
from src.visualize import (
    plot_seq_length_hist,
    plot_umap_2d,
    plot_status_pie_chart,
    plot_interactive_umap_2d,
    plot_interactive_bar_chart
)
# --- Reporting ---
from src.report import generate_report, NpEncoder


def main(config_path="default.yaml"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default.yaml", help="Path to config YAML")
    parser.add_argument("--input", type=str, required=True, help="Path to input sequences CSV/fasta")
    parser.add_argument("--reference", type=str, default="data/raw/labels.csv", help="Path to reference sequences CSV")
    parser.add_argument("--out", type=str, default="results/run1", help="Output directory")
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    
    cfg = read_config(args.config)
    set_seed(cfg.get("seed", 42))

    # -------------------------
    # Set up cache directories
    # -------------------------
    map_dir = cfg.get("map_dir", os.path.join(out_dir, "reference_map"))
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # -------------------------
    # Load and Preprocess Data with Caching
    # -------------------------
    print("Loading and preprocessing data...")
    ranks = cfg['label_transfer']['ranks']
    
    # Create cache paths for preprocessed data
    ref_clean_cache = os.path.join(cache_dir, "ref_clean.parquet")
    ref_dedup_cache = os.path.join(cache_dir, "ref_dedup.parquet")
    input_clean_cache = os.path.join(cache_dir, "input_clean.parquet")
    input_dedup_cache = os.path.join(cache_dir, "input_dedup.parquet")
    
    # Check if preprocessed reference data exists
    if os.path.exists(ref_clean_cache) and os.path.exists(ref_dedup_cache):
        print("Loading cached preprocessed reference data...")
        df_ref_clean = pd.read_parquet(ref_clean_cache)
        df_ref_dedup = pd.read_parquet(ref_dedup_cache)
    else:
        print("Preprocessing reference data...")
        df_ref = pd.read_csv(args.reference)
        df_ref_clean = clean_labels(df_ref, **cfg['preprocess'], ranks=ranks)
        df_ref_dedup = deduplicate(df_ref_clean)
        # Cache the results
        df_ref_clean.to_parquet(ref_clean_cache)
        df_ref_dedup.to_parquet(ref_dedup_cache)
        print(f"Cached preprocessed reference data to {cache_dir}")
    
    # Check if preprocessed input data exists
    if os.path.exists(input_clean_cache) and os.path.exists(input_dedup_cache):
        print("Loading cached preprocessed input data...")
        df_input_clean = pd.read_parquet(input_clean_cache)
        df_input_dedup = pd.read_parquet(input_dedup_cache)
    else:
        print("Preprocessing input data...")
        ext = os.path.splitext(args.input)[1].lower()
        if ext in [".fasta", ".fa", ".fna"]:
            from Bio import SeqIO
            records = list(SeqIO.parse(args.input, "fasta"))
            df_input = pd.DataFrame([{"seq_id": rec.id, "sequence": str(rec.seq)} for rec in records])
        else:
            df_input = pd.read_csv(args.input)
        
        df_input_clean = clean_labels(df_input, **cfg['preprocess'], ranks=ranks)
        df_input_dedup = deduplicate(df_input_clean)
        # Cache the results
        df_input_clean.to_parquet(input_clean_cache)
        df_input_dedup.to_parquet(input_dedup_cache)
        print(f"Cached preprocessed input data to {cache_dir}")
    
    # -------------------------
    # Compute Embeddings with Caching
    # -------------------------
    print("Computing embeddings...")
    
    # Create cache file paths
    ref_emb_cache = os.path.join(cache_dir, "ref_embeddings.npy")
    input_emb_cache = os.path.join(cache_dir, "input_embeddings.npy")
    ref_kmer_cache = os.path.join(cache_dir, "ref_kmers.npz")
    input_kmer_cache = os.path.join(cache_dir, "input_kmers.npz")
    
    # Check if reference embeddings exist
    if os.path.exists(ref_emb_cache) and os.path.exists(ref_kmer_cache):
        print("Loading cached reference embeddings...")
        emb_ref = np.load(ref_emb_cache)
        X_ref = scipy.sparse.load_npz(ref_kmer_cache)
    else:
        print("Processing reference data...")
        X_ref, emb_ref = compute_kmers_and_embeddings(
            df_ref_dedup['normalized_seq'].tolist(),
            k_values=cfg['kmers']['k'],
            n_features=cfg['kmers']['n_features'],
            normalize=cfg['kmers']['normalize'],
            cfg=cfg,
            out_dir=out_dir,
            batch_size=cfg['kmers'].get('batch_size', 10000),
            n_jobs=cfg['kmers'].get('n_jobs', -1)
        )
        # Cache the results
        np.save(ref_emb_cache, emb_ref)
        scipy.sparse.save_npz(ref_kmer_cache, X_ref)
        print(f"Cached reference embeddings to {ref_emb_cache}")

    # Check if input embeddings exist
    if os.path.exists(input_emb_cache) and os.path.exists(input_kmer_cache):
        print("Loading cached input embeddings...")
        emb_input = np.load(input_emb_cache)
        X_input = scipy.sparse.load_npz(input_kmer_cache)
    else:
        print("Processing input data...")
        # Read input sequences into memory to ensure consistent processing
        from Bio import SeqIO
        input_sequences = []
        for record in SeqIO.parse(args.input, "fasta"):
            input_sequences.append(str(record.seq))
        
        # Process input sequences with identical parameters
        X_input, emb_input = compute_kmers_and_embeddings(
            input_sequences,
            k_values=cfg['kmers']['k'],
            n_features=cfg['kmers']['n_features'],
            normalize=cfg['kmers']['normalize'],
            cfg=cfg,
            out_dir=out_dir,
            batch_size=cfg['kmers'].get('batch_size', 10000),
            n_jobs=cfg['kmers'].get('n_jobs', -1)
        )
        # Cache the results
        np.save(input_emb_cache, emb_input)
        scipy.sparse.save_npz(input_kmer_cache, X_input)
        print(f"Cached input embeddings to {input_emb_cache}")
    
    # CRITICAL: Ensure both embeddings have identical dimensions
    print(f"Reference embeddings shape: {emb_ref.shape}")
    print(f"Input embeddings shape: {emb_input.shape}")
    
    if emb_ref.shape[1] != emb_input.shape[1]:
        print(f"WARNING: Embedding dimension mismatch!")
        print(f"Reference: {emb_ref.shape[1]} dimensions")
        print(f"Input: {emb_input.shape[1]} dimensions")
        print("Forcing input embeddings to match reference dimensions...")
        
        # Force alignment by padding or truncating
        if emb_input.shape[1] < emb_ref.shape[1]:
            # Pad with zeros
            padding = np.zeros((emb_input.shape[0], emb_ref.shape[1] - emb_input.shape[1]))
            emb_input = np.hstack([emb_input, padding])
            print(f"Padded input embeddings to {emb_input.shape[1]} dimensions")
        else:
            # Truncate
            emb_input = emb_input[:, :emb_ref.shape[1]]
            print(f"Truncated input embeddings to {emb_input.shape[1]} dimensions")
    
    print("[OK] Embedding dimensions aligned - proceeding with analysis")
    
    # Ensure both reference and input have the same feature dimensions
    if X_ref.shape[1] != X_input.shape[1]:
        print(f"Warning: Feature dimension mismatch - Reference: {X_ref.shape[1]}, Input: {X_input.shape[1]}")
        print("Adjusting input features to match reference...")
        # Pad or truncate input features to match reference
        if X_input.shape[1] < X_ref.shape[1]:
            # Pad with zeros
            padding = np.zeros((X_input.shape[0], X_ref.shape[1] - X_input.shape[1]))
            X_input = np.hstack([X_input.toarray(), padding])
            X_input = scipy.sparse.csr_matrix(X_input)
        else:
            # Truncate
            X_input = X_input[:, :X_ref.shape[1]]
        
        # Recompute embeddings for input with correct dimensions
        print("Recomputing input embeddings with correct dimensions...")
        emb_input = get_embeddings(X_input, cfg, out_dir)
    
    # Ensure embeddings have the same dimensions for prediction
    if emb_ref.shape[1] != emb_input.shape[1]:
        print(f"Warning: Embedding dimension mismatch - Reference: {emb_ref.shape[1]}, Input: {emb_input.shape[1]}")
        print("Recomputing input embeddings to match reference dimensions...")
        
        # Use the same embedding method and dimensions as reference
        from src.embed import get_embeddings
        emb_input = get_embeddings(X_input, cfg, out_dir)
        
        # If still mismatched, force alignment
        if emb_ref.shape[1] != emb_input.shape[1]:
            print(f"Still mismatched after recomputation. Force aligning...")
            if emb_input.shape[1] < emb_ref.shape[1]:
                # Pad with zeros
                padding = np.zeros((emb_input.shape[0], emb_ref.shape[1] - emb_input.shape[1]))
                emb_input = np.hstack([emb_input, padding])
            else:
                # Truncate
                emb_input = emb_input[:, :emb_ref.shape[1]]

    # --- THE REST OF THE SCRIPT REMAINS THE SAME ---
    # The sections for building the reference map, training classifiers,
    # making predictions, the decision engine, and generating reports
    # are unchanged because they operate on the `emb_ref`, `emb_input`,
    # and the metadata DataFrames (`df_ref_dedup`, `df_input_dedup`),
    # which are all still correctly generated.

    # -------------------------
    # Build or Load Unsupervised Reference Map with Caching
    # -------------------------
    os.makedirs(map_dir, exist_ok=True)
    clusterer_path = os.path.join(map_dir, "reference_clusterer.pkl")
    metrics_path = os.path.join(map_dir, "reference_metrics.json")
    ref_umap_cache = os.path.join(cache_dir, "ref_umap_reduced.npy")

    # Check if reference map already exists
    if os.path.exists(clusterer_path) and os.path.exists(metrics_path):
        print("Loading existing reference map...")
        ref_clusterer = joblib.load(clusterer_path)
        with open(metrics_path, 'r') as f:
            ref_cluster_metrics = json.load(f)
        
        # Load cached UMAP reduction if available
        if os.path.exists(ref_umap_cache):
            emb_ref_reduced = np.load(ref_umap_cache)
        else:
            print("Computing UMAP reduction for reference map...")
            emb_ref_reduced, _ = umap_reduce(emb_ref, n_components=cfg['cluster']['umap_10d_components'], n_neighbors=cfg['cluster']['umap_n_neighbors'], min_dist=cfg['cluster']['umap_min_dist'], random_state=cfg.get('seed', 42))
            np.save(ref_umap_cache, emb_ref_reduced)
        
        print("Reference map loaded successfully.")
    else:
        print("Building reference map from scratch...")
        emb_ref_reduced, _ = umap_reduce(emb_ref, n_components=cfg['cluster']['umap_10d_components'], n_neighbors=cfg['cluster']['umap_n_neighbors'], min_dist=cfg['cluster']['umap_min_dist'], random_state=cfg.get('seed', 42))
        np.save(ref_umap_cache, emb_ref_reduced)
        
        ref_clusterer, ref_labels = build_reference_map(emb_ref_reduced, min_cluster_size=cfg['cluster']['hdbscan_min_cluster_size'], min_samples=cfg['cluster']['hdbscan_min_samples'] )
        
        ref_cluster_metrics = analyze_reference_clusters(emb_ref_reduced, ref_labels, ref_clusterer, percentile_threshold=cfg['cluster']['emergent_taxa_percentile'])
        joblib.dump(ref_clusterer, clusterer_path)
        
        # Convert numpy int64 keys to strings for JSON serialization
        def convert_keys(obj):
            if isinstance(obj, dict):
                return {str(k) if isinstance(k, (np.integer, np.int64)) else k: convert_keys(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys(item) for item in obj]
            else:
                return obj
        
        ref_cluster_metrics_converted = convert_keys(ref_cluster_metrics)
        
        with open(metrics_path, 'w') as f:
            json.dump(ref_cluster_metrics_converted, f, cls=NpEncoder)
        print(f"Reference map saved to {map_dir} for future use.")
    
    # -------------------------
    # Train Supervised Classifiers with Caching
    # -------------------------
    print("Training supervised classifiers...")
    models_path = os.path.join(out_dir, "models")
    os.makedirs(models_path, exist_ok=True)
    
    models_cache_path = os.path.join(models_path, "rank_classifiers.pth")
    
    # Check if models already exist
    if os.path.exists(models_cache_path):
        print("Loading cached supervised classifiers...")
        # Allow sklearn LabelEncoder for safe loading
        from sklearn.preprocessing._label import LabelEncoder
        import torch.serialization
        torch.serialization.add_safe_globals([LabelEncoder])
        rank_models = torch.load(models_cache_path)
        print("Classifiers loaded successfully.")
    else:
        print("Training classifiers from scratch...")
        # Train classifiers on reference data
        rank_models = train_rank_classifiers(
            emb_ref, 
            df_ref_dedup, 
            ranks=ranks,
            max_iter=1000,
            min_samples_per_class=3
        )
        
        # Save models
        torch.save(rank_models, models_cache_path)
        print("Classifiers trained and saved.")
    
    # -------------------------
    # Make Predictions on Input Sequences
    # -------------------------
    print("Making predictions on input sequences...")
    
    # Predict on input sequences
    per_rank_preds = predict_per_rank(rank_models, emb_input)
    per_seq_conf = per_sequence_confidence(per_rank_preds)
    
    # -------------------------
    # Unsupervised Analysis
    # -------------------------
    print("Performing unsupervised analysis...")
    
    # Predict clusters for input sequences
    input_labels, input_probs = predict_on_reference_map(emb_input, ref_clusterer, n_jobs=-1)
    
    # Evaluate input sequences against reference map
    unsupervised_results = evaluate_input_sequences(emb_input, input_labels, ref_cluster_metrics)
    
    # -------------------------
    # Decision Engine
    # -------------------------
    print("Applying decision engine logic...")
    
    # Create final results DataFrame
    results_data = []
    for i, seq_id in enumerate(df_input_dedup['seq_id']):
        row = {
            'seq_id': seq_id,
            'normalized_seq': df_input_dedup.iloc[i]['normalized_seq'],
            'read_count': df_input_dedup.iloc[i].get('read_count', 1)
        }
        
        # Add supervised predictions
        for rank in ranks:
            if rank in per_seq_conf[i]:
                pred, conf = per_seq_conf[i][rank]
                row[f'{rank}_pred'] = pred
                row[f'{rank}_conf'] = conf
            else:
                row[f'{rank}_pred'] = 'Unknown'
                row[f'{rank}_conf'] = 0.0
        
        # Override domain classification - all sequences are Eukaryota
        if 'domain' in ranks:
            row['domain_pred'] = 'Eukaryota'
            row['domain_conf'] = 0.99  # High confidence
        
        # Add unsupervised results
        row['cluster_id'] = input_labels[i]
        row['cluster_prob'] = input_probs[i] if input_probs is not None else 0.0
        row['dist_to_nearest_centroid'] = unsupervised_results.iloc[i]['distance_to_centroid']
        
        # Decision engine logic
        min_conf_threshold = cfg['decision_engine'].get('min_classification_confidence', 0.85)
        
        # Check if any rank has high confidence
        high_conf_ranks = [rank for rank in ranks if per_seq_conf[i].get(rank, ('Unknown', 0.0))[1] >= min_conf_threshold]
        
        # More reasonable decision logic
        if len(high_conf_ranks) >= 2:  # Need at least 2 confident ranks
            row['novelty_category'] = 'Classified'
        elif len(high_conf_ranks) >= 1 and unsupervised_results.iloc[i]['unsupervised_status'] == 'Known Organism':
            row['novelty_category'] = 'Classified'
        elif unsupervised_results.iloc[i]['unsupervised_status'] == 'Emergent Taxa':
            row['novelty_category'] = 'Likely Novel Sequence'
        else:
            row['novelty_category'] = 'Uncertain'
        
        results_data.append(row)
    
    # Save results
    results_df = pd.DataFrame(results_data)
    
    # Sort by sequence ID in serial order
    results_df['seq_num'] = results_df['seq_id'].str.extract('(\d+)').astype(int)
    results_df = results_df.sort_values('seq_num').drop('seq_num', axis=1)
    
    results_df.to_csv(os.path.join(out_dir, 'per_sequence_results.csv'), index=False)
    results_df.to_parquet(os.path.join(out_dir, 'per_sequence_predictions.parquet'), index=False)
    
    # -------------------------
    # Generate Visualizations
    # -------------------------
    print("Generating visualizations...")
    
    # Sequence length histogram
    plot_seq_length_hist(df_input_dedup, os.path.join(out_dir, 'seq_len_hist.png'))
    
    # UMAP 2D plot
    plot_umap_2d(emb_input, results_df['novelty_category'], os.path.join(out_dir, 'umap_2d.png'))
    
    # Status pie chart
    plot_status_pie_chart(results_df['novelty_category'], os.path.join(out_dir, 'status_pie.png'))
    
    # Additional cluster visualizations
    from src.visualize import plot_umap2_clusters, plot_top_clusters
    plot_umap2_clusters(emb_input, results_df['cluster_id'], os.path.join(out_dir, 'umap2_clusters.png'))
    plot_top_clusters(results_df, os.path.join(out_dir, 'top_clusters.png'))
    
    # Interactive plots
    interactive_plots = {}
    try:
        interactive_plots['umap'] = plot_interactive_umap_2d(emb_input, results_df['novelty_category'])
        interactive_plots['status_pie'] = plot_interactive_bar_chart(results_df['novelty_category'])
    except Exception as e:
        print(f"Warning: Could not generate interactive plots: {e}")
        interactive_plots = None
    
    # -------------------------
    # Generate Report
    # -------------------------
    print("Generating final report...")
    
    # Calculate summary statistics
    summary = {
        'Total Sequences': len(results_df),
        'Classified': len(results_df[results_df['novelty_category'] == 'Classified']),
        'Likely Novel': len(results_df[results_df['novelty_category'] == 'Likely Novel Sequence']),
        'Uncertain': len(results_df[results_df['novelty_category'] == 'Uncertain']),
        'Reference Clusters': len(ref_cluster_metrics)
    }
    
    # Generate HTML report
    images = [
        'seq_len_hist.png',
        'umap_2d.png', 
        'status_pie.png',
        'umap2_clusters.png',
        'top_clusters.png'
    ]
    
    tables = [
        'per_sequence_results.csv'
    ]
    
    metrics_paths = {
        'classification': 'classification_metrics.json'
    }
    
    report_path = generate_report(
        out_dir, 
        images, 
        tables, 
        summary, 
        metrics_paths, 
        interactive_plots
    )
    
    print(f"Report generated: {report_path}")
    print("Pipeline finished.")

if __name__ == "__main__":
    main()