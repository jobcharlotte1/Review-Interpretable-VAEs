import os
import pandas as pd
import sys
import numpy as np

def log_run(results_dict, file_path):
    """
    Logs a run to a CSV file. Adds a new row with the provided results dictionary.

    Args:
        results_dict (dict): Dictionary with parameters and metrics (e.g., {'param1': ..., 'metric1': ...}).
        file_path (str): Path to the CSV file.
    """
    # Convert dictionary to one-row DataFrame
    df_new = pd.DataFrame([results_dict])

    if os.path.exists(file_path):
        # Load and append
        df_existing = pd.read_csv(file_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(file_path, index=False)
    else:
        # Create new CSV
        df_new.to_csv(file_path, index=False)
        
    return df_new