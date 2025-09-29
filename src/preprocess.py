# src/preprocess.py
import re
import pandas as pd
import hashlib
from pathlib import Path
from .io import save_joblib
from .kmers import build_kmer_matrix
from Bio import SeqIO

# NOTE: The "from .embed import get_embeddings" line that was here is now GONE from the top.

_comp = str.maketrans('ACGT','TGCA')

def sha1(s: str):
    return hashlib.sha1(s.encode('utf8')).hexdigest()

def normalize_seq(s: str):
    if pd.isna(s):
        return ""
    s = s.strip().upper().replace("U", "T")
    s2 = re.sub(r'[^ACGT]', '', s)
    return s2

# ... (fraction_ambiguous, parse_taxonomy, clean_labels, deduplicate functions are all unchanged) ...

def fraction_ambiguous(original, cleaned):
    if not original:
        return 1.0
    return 1.0 - (len(cleaned) / len(original))

def parse_taxonomy(df, tax_col='taxonomy', ranks=None):
    if ranks is None:
        ranks = ["domain","kingdom","phylum","class","order","family","genus","species"]
    def split_pad(t):
        parts = [p.strip() for p in str(t).split(';')]
        parts = [p if p else "Unknown" for p in parts]
        parts = parts + ["Unknown"]*(len(ranks)-len(parts))
        return parts[:len(ranks)]
    cols = df[tax_col].fillna("Unknown").map(split_pad).tolist()
    tax_df = pd.DataFrame(cols, columns=ranks, index=df.index)
    return pd.concat([df.reset_index(drop=True), tax_df.reset_index(drop=True)], axis=1)

def clean_labels(df, seq_col='sequence', taxonomy_col='taxonomy', organism_col='organism_name',
                 max_n_frac=0.05, min_seq_len=50, ranks=None):
    # ... (function body is unchanged)
    df = df.copy()
    df['orig_seq'] = df[seq_col].astype(str)
    df['normalized_seq'] = df['orig_seq'].map(normalize_seq)
    df['orig_len'] = df['orig_seq'].map(len)
    df['clean_len'] = df['normalized_seq'].map(len)
    df['frac_ambig'] = df.apply(lambda r: fraction_ambiguous(r['orig_seq'], r['normalized_seq']), axis=1)
    df = df[df['clean_len'] >= min_seq_len].reset_index(drop=True)
    df = df[df['frac_ambig'] <= max_n_frac].reset_index(drop=True)
    df['seq_id'] = [f"SEQ_{i+1}" for i in range(len(df))]
    if 'read_count' not in df.columns:
        df['read_count'] = 1
    if taxonomy_col in df.columns:
        df = parse_taxonomy(df, tax_col=taxonomy_col, ranks=ranks)
    if organism_col in df.columns:
        df['organism_name'] = df[organism_col]
    else:
        df['organism_name'] = df.get('organism_name', None)
    return df

def deduplicate(df, seq_col='normalized_seq'):
    # ... (function body is unchanged)
    agg = df.groupby(seq_col, as_index=False).agg({
        'seq_id': 'first',
        'read_count': 'sum',
        'organism_name': lambda x: x.dropna().iloc[0] if len(x.dropna())>0 else None,
        'clean_len': 'first',  # Preserve clean_len column
        'orig_len': 'first',   # Preserve orig_len column
        'frac_ambig': 'first'  # Preserve frac_ambig column
    })
    tax_cols = [c for c in df.columns if c in ["domain","kingdom","phylum","class","order","family","genus","species"]]
    for t in tax_cols:
        rep = df.groupby(seq_col)[t].agg(lambda x: x.dropna().iloc[0] if len(x.dropna())>0 else "Unknown").reset_index()
        agg = agg.merge(rep, on=seq_col, how='left')
    return agg

def fasta_batch_generator(fasta_path, batch_size=10000):
    """
    Generator that yields batches of sequences from a FASTA file.
    Memory-efficient for large files.
    """
    batch = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        batch.append(str(record.seq))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    # Yield the remaining sequences in the last batch
    if batch:
        yield batch

# In src/preprocess.py

# In src/preprocess.py

def compute_kmers_and_embeddings(data_input, k_values=[7], n_features=65536, normalize=True, cfg=None, out_dir=None, batch_size=10000, n_jobs=-1):
    """
    Compute k-mer matrix and embeddings. Handles either a file path (for batching)
    or a list/Series of sequences (for smaller, in-memory data).
    """
    from .embed import get_embeddings
    from .kmers import build_kmer_matrix # Ensure this is imported
    from scipy.sparse import vstack
    import os # Add os import for path check

    print(f"Computing multi-k-mer matrix for k={k_values}...")
    
    batch_matrices = []
    
    # Check if input is a file path (string) or a list of sequences
    if isinstance(data_input, str) and os.path.exists(data_input):
        print("Input is a file path. Processing in batches...")
        seq_generator = fasta_batch_generator(data_input, batch_size=batch_size)
        
        for i, seq_batch in enumerate(seq_generator):
            print(f"  - Processing batch {i+1}...")
            normalized_batch = [normalize_seq(s) for s in seq_batch]
            X_batch_sparse = build_kmer_matrix(
                normalized_batch, k_values=k_values, n_features=n_features, normalize_rows=normalize, n_jobs=n_jobs
            )
            batch_matrices.append(X_batch_sparse)
    else: # Assumes input is a list or pandas Series of sequences
        print("Input is a list of sequences. Processing all at once...")
        normalized_sequences = [normalize_seq(s) for s in data_input]
        
        # --- OPTIMIZED: Use configuration parameters for better performance ---
        X_sparse = build_kmer_matrix(
            normalized_sequences, 
            k_values=k_values, 
            n_features=n_features, 
            normalize_rows=normalize,
            n_jobs=n_jobs,
            batch_size=batch_size
        )
        batch_matrices.append(X_sparse)

    print("All data processed. Stacking matrices...")
    if not batch_matrices:
        raise ValueError("No sequences were processed. Check your input data.")
    X_sparse = vstack(batch_matrices) if len(batch_matrices) > 1 else batch_matrices[0]
    
    print("Computing embeddings for the full matrix...")
    emb = get_embeddings(X_sparse, cfg, out_dir)
    
    return X_sparse, emb