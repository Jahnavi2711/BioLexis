# src/evaluate.py
import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classifiers(models, ranks, X_val, df_val, out_dir):
    """
    Evaluate per-rank classifiers on validation data.
    Unseen labels in validation set are mapped to 'Unknown'.
    Skips ranks with no trained classifier.
    """
    metrics = {}
    os.makedirs(out_dir, exist_ok=True)

    for r in ranks:
        clf, le = models.get(r, (None, None))
        if clf is None or le is None:
            print(f"[WARN] No classifier for rank '{r}', skipping evaluation.")
            continue

        y_true = df_val[r].fillna("Unknown").values
        y_true_safe = np.array([label if label in le.classes_ else "Unknown" for label in y_true])

        if 'Unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'Unknown')

        try:
            y_true_enc = le.transform(y_true_safe)
        except ValueError:
            # fallback if all labels unseen
            y_true_enc = np.zeros(len(y_true_safe), dtype=int)

        if clf is not None:
            y_pred_enc = clf.predict(X_val)
            y_pred = le.inverse_transform(y_pred_enc)
        else:
            y_pred_enc = np.zeros(len(y_true_safe), dtype=int)
            y_pred = np.array(["Unknown"] * len(y_true_safe))

        acc = accuracy_score(y_true_enc, y_pred_enc)
        prec, recall, f1, _ = precision_recall_fscore_support(y_true_enc, y_pred_enc, average='macro', zero_division=0)
        weighted_f1 = precision_recall_fscore_support(y_true_enc, y_pred_enc, average='weighted', zero_division=0)[2]

        metrics[r] = {
            "accuracy": acc,
            "macro_f1": f1,
            "weighted_f1": weighted_f1
        }

        if r in ['genus', 'species']:
            plot_confusion_matrix(y_true_safe, y_pred,
                                  title=f"Confusion Matrix - {r}",
                                  out_path=os.path.join(out_dir, f"confusion_matrix_{r}.png"))

    with open(os.path.join(out_dir, "classification_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def plot_confusion_matrix(y_true, y_pred, title, out_path, max_labels=30):
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if len(labels) > max_labels:
        top_idx = np.argsort(-cm.sum(axis=1))[:max_labels]
        cm = cm[top_idx][:, top_idx]
        labels = labels[top_idx]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=False, cbar=True)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate_clustering(emb, cluster_df, true_labels=None, out_dir=None):
    from sklearn.metrics import silhouette_score
    results = {}

    mask = cluster_df['cluster_id'] != -1
    if mask.sum() > 1:
        results['silhouette_score'] = float(silhouette_score(emb[mask], cluster_df['cluster_id'][mask]))
    else:
        results['silhouette_score'] = None

    results['noise_ratio'] = float((cluster_df['cluster_id'] == -1).sum() / len(cluster_df))

    if true_labels is not None:
        cluster_purity = compute_purity(cluster_df['cluster_id'], true_labels)
        results['purity'] = cluster_purity

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "clustering_metrics.json"), 'w') as f:
            json.dump(results, f, indent=2)

    return results


def compute_purity(cluster_labels, true_labels):
    df = pd.DataFrame({'cluster': cluster_labels, 'true': true_labels})
    total = len(df)
    purity_sum = 0
    for c in df['cluster'].unique():
        if c == -1:
            continue
        subset = df[df['cluster'] == c]
        top_count = subset['true'].value_counts().iloc[0]
        purity_sum += top_count
    return round(purity_sum / total, 4)
