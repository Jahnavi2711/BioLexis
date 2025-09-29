# src/clusters.py
import umap
import hdbscan
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# --- UNCHANGED FUNCTION ---
def umap_reduce(X, n_components=10, n_neighbors=30, min_dist=0.1, random_state=0):
    """Reduces dimensionality of embeddings using UMAP."""
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=random_state)
    Z = reducer.fit_transform(X)
    return Z, reducer

# --- NEW FUNCTIONS FOR REFERENCE MAP BUILDING ---

# In src/cluster.py

def build_reference_map(emb_ref, min_cluster_size=50, min_samples=5,):
    """
    Builds the HDBSCAN reference map by clustering the reference embeddings.
    """
    print(f"Building HDBSCAN reference map from {len(emb_ref)} reference sequences...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,  # <<< THIS LINE IS THE FIX
        core_dist_n_jobs=-1    # Use all available CPU cores to speed up
    )
    labels = clusterer.fit_predict(emb_ref)
    print(f"Reference map built. Found {len(np.unique(labels)) - 1} clusters.")
    return clusterer, labels

def analyze_reference_clusters(emb_ref, labels_ref, clusterer_ref, percentile_threshold=98):
    """
    Calculates centroids and distance thresholds for each cluster in the reference map.
    
    Returns a dictionary of metrics for each valid cluster.
    """
    print("Calculating centroids and distance thresholds for reference clusters...")
    cluster_metrics = {}
    valid_cluster_ids = np.unique(labels_ref[labels_ref != -1])
    
    # Calculate centroids manually for HDBSCAN clusters
    centroids = []
    centroid_map = {}
    for i, cluster_id in enumerate(valid_cluster_ids):
        member_mask = (labels_ref == cluster_id)
        member_embeddings = emb_ref[member_mask]
        # Calculate centroid as mean of cluster members
        centroid = np.mean(member_embeddings, axis=0)
        centroids.append(centroid)
        centroid_map[cluster_id] = i
    
    centroids = np.array(centroids)
    
    for cluster_id, centroid_idx in centroid_map.items():
        member_mask = (labels_ref == cluster_id)
        member_embeddings = emb_ref[member_mask]
        cluster_centroid = centroids[centroid_idx].reshape(1, -1)
        
        # Calculate distances for all reference members of this cluster
        distances = cdist(member_embeddings, cluster_centroid).flatten()
        
        # Calculate the distance threshold from reference data
        distance_threshold = np.percentile(distances, percentile_threshold)
        
        cluster_metrics[cluster_id] = {
            'centroid': centroids[centroid_idx],
            'distance_threshold': distance_threshold
        }
    print("Reference cluster metrics calculated.")
    return cluster_metrics

# --- NEW FUNCTIONS FOR PREDICTING INPUT SEQUENCES ---



# ... (other functions) ...

def predict_on_reference_map(emb_input, clusterer_ref, n_jobs=-1):
    """
    Predicts cluster assignments for input sequences on the pre-built reference map.
    """
    print(f"Predicting cluster assignments...")
    
    # Check dimensions before prediction
    print(f"Input embeddings shape: {emb_input.shape}")
    print(f"Reference clusterer was trained on embeddings with shape: {clusterer_ref._raw_data.shape}")
    
    # Ensure dimensions match
    if emb_input.shape[1] != clusterer_ref._raw_data.shape[1]:
        print(f"WARNING: Dimension mismatch detected!")
        print(f"Input embeddings: {emb_input.shape[1]} dimensions")
        print(f"Reference embeddings: {clusterer_ref._raw_data.shape[1]} dimensions")
        print("Attempting to fix dimension mismatch...")
        
        # Pad or truncate input embeddings to match reference
        if emb_input.shape[1] < clusterer_ref._raw_data.shape[1]:
            # Pad with zeros
            padding = np.zeros((emb_input.shape[0], clusterer_ref._raw_data.shape[1] - emb_input.shape[1]))
            emb_input = np.hstack([emb_input, padding])
            print(f"Padded input embeddings to {emb_input.shape[1]} dimensions")
        else:
            # Truncate
            emb_input = emb_input[:, :clusterer_ref._raw_data.shape[1]]
            print(f"Truncated input embeddings to {emb_input.shape[1]} dimensions")
    
    # HDBSCAN approximate_predict doesn't support n_jobs parameter
    labels, probs = hdbscan.approximate_predict(clusterer_ref, emb_input)
    
    print("Prediction complete.")
    return labels, probs

def evaluate_input_sequences(emb_input, labels_input, cluster_metrics):
    """
    Assigns an unsupervised status to each input sequence based on the reference map.
    """
    print("Assigning unsupervised status to input sequences...")
    num_samples = len(labels_input)
    results_df = pd.DataFrame({
        'cluster_id': labels_input,
        'unsupervised_status': ['Outlier'] * num_samples,
        'distance_to_centroid': [np.nan] * num_samples
    })

    for i in range(num_samples):
        cluster_id = labels_input[i]
        
        # Skip outliers
        if cluster_id == -1:
            continue
            
        # Check if the predicted cluster exists in our reference metrics
        if cluster_id not in cluster_metrics:
            results_df.loc[i, 'unsupervised_status'] = 'Outlier' # Treat as outlier if cluster is unknown
            continue

        # Get pre-computed metrics for this reference cluster
        metrics = cluster_metrics[cluster_id]
        centroid = metrics['centroid'].reshape(1, -1)
        threshold = metrics['distance_threshold']
        
        # Calculate distance of the input point to the reference centroid
        distance = cdist(emb_input[i].reshape(1, -1), centroid)[0][0]
        results_df.loc[i, 'distance_to_centroid'] = distance
        
        # Compare to the pre-computed threshold
        if distance <= threshold:
            results_df.loc[i, 'unsupervised_status'] = 'Known Organism'
        else:
            results_df.loc[i, 'unsupervised_status'] = 'Emergent Taxa'
            
    print("Unsupervised status assigned.")
    return results_df



