# src/kmers.py
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import normalize
import numpy as np
from joblib import Parallel, delayed
import scipy.sparse
from tqdm import tqdm

# --- revcomp, canonical_kmer, seq_to_multi_kmer_counts are unchanged ---

_comp = str.maketrans('ACGT','TGCA')

def revcomp(s):
    return s.translate(_comp)[::-1]

def canonical_kmer(kmer):
    rc = revcomp(kmer)
    # Ensure both are strings for comparison
    kmer = str(kmer)
    rc = str(rc)
    return kmer if kmer <= rc else rc

def seq_to_multi_kmer_counts(seq, k_values):
    counts = {}
    L = len(seq)
    for k in k_values:
        if L < k:
            continue
        for i in range(L - k + 1):
            kmer = seq[i:i+k]
            if any(c not in 'ACGT' for c in kmer):
                continue
            ck = canonical_kmer(kmer)
            k_prefixed_ck = f"{k}_{ck}"
            counts[k_prefixed_ck] = counts.get(k_prefixed_ck, 0) + 1
    return counts

# OPTIMIZED: This function now uses batch processing to be memory-safe and faster
def build_kmer_matrix(sequences, k_values, n_features=65536, normalize_rows=True, dtype='float32', n_jobs=-1, batch_size=10000):
    """
    Builds a sparse matrix of k-mer counts in parallel using memory-safe batch processing.
    """
    hasher = FeatureHasher(n_features=n_features, input_type='dict', alternate_sign=False, dtype=np.float32)
    
    # Convert sequences to a list to calculate total number
    sequences = list(sequences)
    n_sequences = len(sequences)
    
    matrix_chunks = []
    
    print(f"Generating k-mer counts in batches (size={batch_size})...")
    
    # Process the data in batches
    for i in tqdm(range(0, n_sequences, batch_size), desc="Processing Batches"):
        # Get a small batch of sequences
        batch_seqs = sequences[i:i + batch_size]
        
        # Process this small batch in parallel with optimized backend
        tasks = (delayed(seq_to_multi_kmer_counts)(s, k_values=k_values) for s in batch_seqs)
        # Use 'loky' backend for better performance on multi-core systems
        dicts_batch = Parallel(n_jobs=n_jobs, backend="loky", prefer="threads")(tasks)
        
        # Transform the small batch of dicts into a sparse matrix chunk
        X_chunk = hasher.transform(dicts_batch)
        matrix_chunks.append(X_chunk)
        
    # Combine all the small chunks into one large sparse matrix
    print("Combining batch results into final matrix...")
    X = scipy.sparse.vstack(matrix_chunks, format='csr')
    
    if normalize_rows:
        print("Normalizing matrix rows...")
        X = normalize(X, norm='l2', axis=1)
        
    return X