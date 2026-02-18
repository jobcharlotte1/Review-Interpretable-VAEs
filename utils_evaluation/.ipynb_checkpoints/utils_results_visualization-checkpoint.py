import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.colors as pc
import plotly.express as px
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

def compute_model_split_statistics(df_scores, required_columns, columns_grouped_by):
    # Ensure required columns exist
    missing = required_columns - set(df_scores.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # Metrics to aggregate
    metrics = [
        'mse_score', 'corr', 'ari', 'nmi',
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
    ]

    splits = ['train', 'val', 'test']
    models = ['', '_rf', '_xg']  # '' = latent metrics, rf / xg = classifiers

    # Aggregation functions
    agg_fns = ['mean', 'std', 'median',
               lambda x: x.quantile(0.25),
               lambda x: x.quantile(0.75)]

    agg_dict = {}

    for metric in metrics:
        for split in splits:
            for model in models:
                col = f"{metric}_{split}{model}".strip("_")
                if col in df_scores.columns:
                    agg_dict[col] = agg_fns

    # Perform aggregation
    stats = (
        df_scores
        .groupby(columns_grouped_by)
        .agg(agg_dict)
    )

    # Rename columns
    new_columns = []
    for col, stat in stats.columns:
        base = col.replace("score_", "")
        if stat == '<lambda_0>':
            stat = 'q25'
        elif stat == '<lambda_1>':
            stat = 'q75'
        new_columns.append(f"{base}_{stat}")

    stats.columns = new_columns

    # Reset index for a flat DataFrame
    stats = stats.reset_index()

    return stats


def plot_metric_distribution(
    df: pd.DataFrame,
    metric: str,
    split: str = "test",
    classifier_suffix: str = None,
    y_range: tuple = None
):
    """
    Plots the distribution of a metric across datasets and models using vertical boxplots.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing metrics. Must have columns: 'model', 'dataset_name', and metric-specific columns.
    metric : str
        Base name of the metric (e.g., 'accuracy', 'f1', 'roc_auc').
    split : str, default 'test'
        Split name to select (e.g., 'train' or 'test').
    classifier_suffix : str, optional
        Optional classifier suffix (e.g., 'rf', 'xg') — will be appended with underscore automatically.
    y_range : tuple, optional
        Tuple of (min, max) to set y-axis limits.
    """
    # Add underscore before classifier suffix if provided
    suffix = f"_{classifier_suffix}" if classifier_suffix else ""
    col_name = f"{metric}_{split}{suffix}"

    if col_name not in df.columns:
        raise KeyError(f"Metric column '{col_name}' not found in DataFrame.")

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="dataset_name",
        y=col_name,
        hue="model",
        data=df,
        palette="Set2"
    )

    plt.xlabel("Dataset")
    plt.ylabel(f"{metric.replace('_',' ').title()} ({split})" + (f" [{classifier_suffix}]" if classifier_suffix else ""))
    plt.title(f"{metric.replace('_',' ').title()} Distribution Across Models")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')

    if y_range is not None:
        plt.ylim(y_range)

    plt.tight_layout()
    plt.show()
    
def plot_metric_distribution_plotly(
    df: pd.DataFrame,
    name_column_labels:str,
    metric: str,
    split: str = "test",
    classifier_suffix: str = None,
    y_range: tuple = None,
    colors: str = None,
    save_path: str = None
):
    """
    Plotly vertical boxplot of a metric across datasets and models with:
    - consistent pastel colors per model
    - no points
    - hover showing model name and value
    """
    suffix = f"_{classifier_suffix}" if classifier_suffix else ""
    col_name = f"{metric}_{split}{suffix}"

    if col_name not in df.columns:
        raise KeyError(f"Metric column '{col_name}' not found in DataFrame.")

    fig = go.Figure()

    datasets = df["dataset_name"].unique()
    models = df[name_column_labels].unique()

    # pastel color palette
    if colors == None:
        colors = px.colors.qualitative.Pastel
    color_map = {model: colors[i % len(colors)] for i, model in enumerate(models)}

    for model in models:
        for dataset in datasets:
            values = df.loc[
                (df[name_column_labels] == model) &
                (df["dataset_name"] == dataset),
                col_name
            ]
            if values.empty:
                continue

            fig.add_trace(go.Box(
                y=values,
                x=[dataset] * len(values),
                name=model,
                boxpoints=False,  # no individual points
                marker=dict(color=color_map[model]),
                legendgroup=model,
                offsetgroup=model,
                showlegend=(dataset == datasets[0]),
                hovertemplate=(
                    "Model: %{name}<br>" +  # model name
                    "Value: %{y}<extra></extra>"  # show value, remove secondary info
                )
            ))

    fig.update_layout(
        title=f"{metric.replace('_',' ').title()} Distribution Across Models",
        xaxis_title="Dataset",
        yaxis_title=f"{metric.replace('_',' ').title()} ({split})" + (f" [{classifier_suffix}]" if classifier_suffix else ""),
        boxmode="group",
        template="simple_white",
        font=dict(size=14)
    )

    if y_range:
        fig.update_yaxes(range=y_range)
        
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if classifier_suffix == None:
            fig.write_html(save_path + f'fig_{name_column_labels}_{metric}_{split}_distribution.html')
        else:
            fig.write_html(save_path + f'fig_{name_column_labels}_{metric}_{split}_{classifier_suffix}_distribution.html')
     
    fig.show()