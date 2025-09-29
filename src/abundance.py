# src/abundance.py
import pandas as pd
import numpy as np

def aggregate_abundance(df, group_by_col, read_count_col='read_count'):
    """
    Aggregates read counts from a single dataframe to calculate abundance.

    Args:
        df (pd.DataFrame): A DataFrame that must contain the columns specified by
                           group_by_col and read_count_col. Can optionally
                           contain a 'sample_id' column for per-sample analysis.
        group_by_col (str): The name of the column containing the assignments to group by
                            (e.g., 'final_assignment').
        read_count_col (str): The name of the column containing the raw read counts.

    Returns:
        pd.DataFrame: A DataFrame with absolute and relative abundance for each group,
                      sorted by abundance.
    """
    # Check that the necessary columns exist in the dataframe
    required_cols = [group_by_col, read_count_col]
    if 'sample_id' in df.columns:
        required_cols.append('sample_id')

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input DataFrame is missing the required column: '{col}'")
    
    # Handle per-sample abundance if sample_id is present
    if 'sample_id' in df.columns:
        grouped = df.groupby(['sample_id', group_by_col])[read_count_col].sum().reset_index()
        
        # Calculate relative abundance within each sample
        grouped['rel_abundance'] = grouped.groupby('sample_id')[read_count_col].transform(lambda s: s / s.sum())
    
    # Handle overall abundance
    else:
        grouped = df.groupby(group_by_col)[read_count_col].sum().reset_index()
        total_reads = grouped[read_count_col].sum()
        grouped['rel_abundance'] = grouped[read_count_col] / (total_reads if total_reads > 0 else 1)

    # Sort the final table by abundance for clarity
    final_df = grouped.sort_values(by=read_count_col, ascending=False).reset_index(drop=True)

    return final_df