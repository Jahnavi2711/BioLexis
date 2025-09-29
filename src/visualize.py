# src/visualize.py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio

# Set default theme for plotly
pio.templates.default = "plotly_white"

# --- STATIC PLOTS (Unchanged) ---
def plot_seq_length_hist(df, out_path, bins=100):
    """Plots a static histogram of sequence lengths."""
    # Defensive programming: check for clean_len column, fallback to calculating from normalized_seq
    if 'clean_len' in df.columns:
        lengths = df['clean_len'].values
    elif 'normalized_seq' in df.columns:
        lengths = df['normalized_seq'].str.len().values
    else:
        raise ValueError("DataFrame must contain either 'clean_len' or 'normalized_seq' column for length histogram")
    
    plt.figure(figsize=(6,4))
    plt.hist(lengths, bins=bins)
    plt.title("Sequence Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_umap_2d(Z2, labels, out_path, markersize=4, alpha=0.7):
    """Creates a static 2D scatter plot of UMAP embeddings."""
    plt.figure(figsize=(6,6))
    plt.scatter(Z2[:,0], Z2[:,1], s=markersize, c=labels, cmap='viridis')
    plt.title("UMAP 2D Embedding Colored by Cluster ID")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# --- MODIFIED INTERACTIVE PLOTS (now return HTML) ---

def plot_status_pie_chart(df_results):
    """
    MODIFIED: Creates an interactive pie chart and returns its HTML source code.
    """
    print("Creating interactive pie chart HTML...")
    
    # Defensive programming: handle different column names
    status_col = None
    for col in ['final_status', 'novelty_category', 'status']:
        if col in df_results.columns:
            status_col = col
            break
    
    if status_col is None:
        return "<div>No status column found in results data. Available columns: " + ", ".join(df_results.columns) + "</div>"
    
    status_counts = df_results[status_col].value_counts().reset_index()
    status_counts.columns = [status_col, 'count']
    
    fig = px.pie(status_counts, 
                 names=status_col, 
                 values='count',
                 title='Final Status Distribution',
                 hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    # Return HTML string instead of writing to file
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_interactive_umap_2d(df_plot):
    """
    MODIFIED: Creates an interactive UMAP plot and returns its HTML source code.
    """
    print("Creating interactive UMAP plot HTML...")
    
    # Defensive programming: check for required columns
    required_cols = ['UMAP1', 'UMAP2', 'cluster_id']
    missing_cols = [col for col in required_cols if col not in df_plot.columns]
    if missing_cols:
        return f"<div>Missing required columns for UMAP plot: {', '.join(missing_cols)}. Available columns: {', '.join(df_plot.columns)}</div>"
    
    df_plot['cluster_id'] = df_plot['cluster_id'].astype(str)

    # Build hover_data dynamically based on available columns
    hover_cols = []
    for col in ['seq_id', 'final_status', 'novelty_category', 'species_pred', 'genus_pred']:
        if col in df_plot.columns:
            hover_cols.append(col)

    fig = px.scatter(df_plot, 
                     x='UMAP1', 
                     y='UMAP2', 
                     color='cluster_id',
                     title='Interactive UMAP Colored by Predicted Cluster ID',
                     labels={'cluster_id': 'Cluster ID'},
                     hover_data=hover_cols)
    
    fig.update_layout(legend_title_text='Cluster ID')
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    # Return HTML string instead of writing to file
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_interactive_bar_chart(abund_df, top_n=20):
    """
    MODIFIED: Creates an interactive bar chart and returns its HTML source code.
    """
    print("Creating interactive bar chart HTML...")
    if abund_df.empty:
        return "<div>Abundance dataframe is empty. No plot generated.</div>"
    
    # Defensive programming: check for required columns
    if 'read_count' not in abund_df.columns:
        return f"<div>Missing 'read_count' column. Available columns: {', '.join(abund_df.columns)}</div>"
    
    # Find assignment column
    assignment_col = None
    for col in ['final_assignment', 'assignment', 'species_pred', 'genus_pred']:
        if col in abund_df.columns:
            assignment_col = col
            break
    
    if assignment_col is None:
        return f"<div>No assignment column found. Available columns: {', '.join(abund_df.columns)}</div>"
        
    g = abund_df.sort_values('read_count', ascending=False).head(top_n)

    # Build hover_data dynamically
    hover_cols = []
    for col in ['rel_abundance', 'abundance', 'count']:
        if col in g.columns:
            hover_cols.append(col)

    fig = px.bar(g, 
                 x=assignment_col, 
                 y='read_count',
                 title=f'Top {top_n} Assignments by Read Count',
                 hover_data=hover_cols,
                 labels={assignment_col: 'Assignment', 'read_count': 'Read Count'})

    fig.update_layout(xaxis_title="", xaxis_tickangle=-60)
    # Return HTML string instead of writing to file
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_umap2_clusters(embeddings, labels, out_path):
    """Plot UMAP 2D with cluster colors."""
    import umap
    import matplotlib.pyplot as plt
    
    # Reduce to 2D
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Handle outlier labels (-1) by converting to a special value
    plot_labels = labels.copy()
    unique_labels = np.unique(plot_labels)
    
    # Create a color map that handles outliers
    if -1 in unique_labels:
        # Convert -1 to a high number for coloring
        non_outlier_labels = unique_labels[unique_labels != -1]
        if len(non_outlier_labels) > 0:
            max_label = np.max(non_outlier_labels)
            plot_labels[plot_labels == -1] = max_label + 1
        else:
            # All labels are -1, just use 0
            plot_labels[plot_labels == -1] = 0
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=plot_labels, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter)
    plt.title("UMAP 2D Clusters")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_top_clusters(results_df, out_path, top_n=10):
    """Plot top clusters by sequence count."""
    # Defensive programming: check for cluster_id column
    if 'cluster_id' not in results_df.columns:
        print(f"Warning: 'cluster_id' column not found in results_df. Available columns: {', '.join(results_df.columns)}")
        return
    
    cluster_counts = results_df['cluster_id'].value_counts().head(top_n)
    
    plt.figure(figsize=(12, 6))
    cluster_counts.plot(kind='bar')
    plt.title(f"Top {top_n} Clusters by Sequence Count")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Sequences")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()