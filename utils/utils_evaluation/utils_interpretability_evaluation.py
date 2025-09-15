import os
import sys
import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr


def cosine_distances_between_columns(df1, df2):
    distances = {}
    for col in df1.columns:
        vec1 = df1[col].values
        vec2 = df2[col].values
        distances[col] = cosine(vec1, vec2)
    return pd.DataFrame([distances])

def save_or_append_df(new_row_df, file_path, pathway_perturbated):
    """
    Save a one-row DataFrame to a CSV file with index set to pathway_perturbated.
    If the file exists, append the new row. Otherwise, create the file.

    Parameters:
    - new_row_df: pd.DataFrame with one row.
    - file_path: str, path to the CSV file.
    - pathway_perturbated: str, label to use as the index for the row.
    """
    # Set the index to the pathway name
    new_row_df.index = [pathway_perturbated]

    if not os.path.exists(file_path):
        # Save with index if file doesn't exist
        new_row_df.to_csv(file_path, index=True)
        print(f"Created new file at {file_path}")
    else:
        # Load existing file with index
        existing_df = pd.read_csv(file_path, index_col=0)

        if list(existing_df.columns) != list(new_row_df.columns):
            raise ValueError("Column mismatch between new data and existing file.")

        # Concatenate
        combined_df = pd.concat([existing_df, new_row_df])

        # Save updated DataFrame
        combined_df.to_csv(file_path, index=True)
        print(f"Appended row with index '{pathway_perturbated}' to {file_path}")
        return combined_df


def plot_distribution_distances(result_ordered, top_pathways, pathway_selected, n, path_save_fig):
    top_pathways = list(result_ordered.columns[:n])
    colors = ['darkblue' if i < n else 'lightblue' for i in range(len(result_ordered.columns))]
    
    x = range(len(result_ordered.columns))
    y = result_ordered.iloc[0].values

    plt.figure(figsize=(15, 6)) 

    # Plot the line with a single color
    plt.plot(x, y, color='gray', linewidth=1, label='Distance')

    # Plot the markers individually with their colors
    for xi, yi, ci in zip(x, y, colors):
        plt.scatter(xi, yi, color=ci)

    # Create custom legend handles for the colors
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='darkblue', label=f"First {n} modified pathways: \n " + " \n ".join(top_pathways[:n])),
        mpatches.Patch(color='lightblue', label='Other pathways')
    ]

    plt.legend(handles=legend_handles)

    plt.xlabel('Pathways')
    plt.ylabel('Cosine distance')
    plt.title(f'{pathway_selected}')
    plt.savefig(path_save_fig+f'distribution_distances_{pathway_selected}.png')
    plt.show()
    
    
def plot_hist_distances(result_ordered, pathway_selected, n, path_save_fig):
    top_pathways = list(result_ordered.columns[:n])
    colors = ['darkblue' if i < n else 'lightblue' for i in range(len(result_ordered.columns))]

    plt.figure(figsize=(20, 5))
    bars = plt.bar(range(len(result_ordered.columns)), result_ordered.iloc[0].values, color=colors, width=2)
    plt.xlabel('Pathways')
    plt.ylabel('Cosine Distance')
    plt.title(f'Distance between original and perturbated embedding for pathway {pathway_selected}')

    # Add labels only to the top pathways bars
    for i, col in enumerate(result_ordered.columns):
        if col in top_pathways:
            bars[i].set_label(f'{col}: {result_ordered.at[pathway_selected, col]:.4f}')

    # Create legend from labeled bars
    plt.legend(title=f"Top {n} distances", loc='upper right', fontsize=8, title_fontsize=9)

    plt.tight_layout()
    plt.savefig(path_save_fig+f'hist_distances_{pathway_selected}.png')
    plt.show()
