# src/label_transfer.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from collections import Counter
import torch  # for saving/loading models

# ----------------------------
# Training per-rank classifiers
# ----------------------------
def train_rank_classifiers(X_train, df_train, ranks, max_iter=1000, min_samples_per_class=3):
    """
    Train one classifier per rank after filtering out rare classes.
    Skips training if only one class remains.
    """
    models = {}

    for r in ranks:
        print(f"\n[INFO] Processing rank: {r}")
        y = df_train[r].fillna("Unknown").values
        class_counts = Counter(y)
        print(f"Original class distribution for {r}: {dict(class_counts)}")
        valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples_per_class]
        print(f"Valid classes (>= {min_samples_per_class} samples): {valid_classes}")

        if len(valid_classes) == 0:
            print(f"[WARNING] No valid classes for rank '{r}', skipping.")
            models[r] = (None, None)
            continue

        mask = np.isin(y, valid_classes)
        X_filtered = X_train[mask]
        y_filtered = y[mask]

        le = LabelEncoder()
        y_enc = le.fit_transform(y_filtered)

        if len(le.classes_) < 2:
            print(f"[WARNING] Only one class remains for rank '{r}' after filtering, skipping training.")
            models[r] = (None, le)
            continue

        base = LogisticRegression(max_iter=max_iter, class_weight='balanced', solver='lbfgs')
        clf = CalibratedClassifierCV(base, cv=3)
        clf.fit(X_filtered, y_enc)

        models[r] = (clf, le)
        print(f"[INFO] Classifier trained for rank '{r}' with {len(le.classes_)} classes.")

    return models

# ----------------------------
# Predict per rank
# ----------------------------
def predict_per_rank(models, X):
    """
    Predict labels and probabilities for each rank.
    Returns:
        dict: rank -> (labels array, probabilities or None)
    """
    out = {}

    for r, (clf, le) in models.items():
        if clf is None:
            n_samples = X.shape[0]
            out[r] = (np.array(["Unknown"] * n_samples), None)
            continue
        if len(le.classes_) == 1:
            label = le.classes_[0]
            n_samples = X.shape[0]
            out[r] = (np.array([label] * n_samples), None)
            continue

        probs = clf.predict_proba(X)
        idx = probs.argmax(axis=1)
        labels = le.inverse_transform(idx)
        out[r] = (labels, probs)

    return out

# ----------------------------
# Per-sequence confidence
# ----------------------------
def per_sequence_confidence(per_seq_preds):
    n = len(next(iter(per_seq_preds.values()))[0])
    per_seq = [{} for _ in range(n)]

    for rank, (labels, probs) in per_seq_preds.items():
        if probs is None:
            for i in range(n):
                per_seq[i][rank] = (labels[i], 0.0)
        else:
            max_conf = probs.max(axis=1)
            for i in range(n):
                per_seq[i][rank] = (labels[i], float(max_conf[i]))

    return per_seq

# ----------------------------
# Cluster labeling
# ----------------------------
def cluster_labeling(cluster_df, per_seq_preds, min_confidence=0.6):
    seq_to_idx = {sid: i for i, sid in enumerate(cluster_df['seq_id'].values)}
    clusters = sorted(cluster_df['cluster_id'].unique())
    out = {}

    for c in clusters:
        members = cluster_df[cluster_df['cluster_id'] == c]['seq_id'].tolist()
        if len(members) == 0:
            out[c] = {}
            continue

        mapping = {}
        for rank, (labels, probs) in per_seq_preds.items():
            member_idx = [seq_to_idx[s] for s in members if s in seq_to_idx]
            if len(member_idx) == 0:
                mapping[rank] = ("Unknown", 0.0)
                continue

            if probs is None:
                votes = Counter(labels[member_idx])
                top_label, cnt = votes.most_common(1)[0]
                mapping[rank] = (top_label, 0.0)
                continue

            pred_labels = labels[member_idx]
            confs = probs[member_idx].max(axis=1)
            votes = Counter(pred_labels)
            top_label, top_cnt = votes.most_common(1)[0]

            mean_conf = (
                float(confs[[i for i, l in enumerate(pred_labels) if l == top_label]].mean())
                if top_cnt > 0 else 0.0
            )

            if mean_conf >= min_confidence and (top_cnt / len(pred_labels)) >= 0.5:
                mapping[rank] = (top_label, mean_conf)
            else:
                mapping[rank] = ("Unknown", mean_conf)

        out[c] = mapping

    return out

# ----------------------------
# Save / Load models
# ----------------------------
def save_models(models, path):
    """
    Save trained models dictionary to a file.
    """
    torch.save(models, path)
    print(f"[INFO] Models saved to {path}")

def load_models(path):
    """
    Load trained models dictionary from a file.
    """
    models = torch.load(path)
    print(f"[INFO] Models loaded from {path}")
    return models
