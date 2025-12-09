import numpy as np
import pandas as pd


def precision_genes_score(list_genes_important, list_genes_true):
    set_intersection = set(list_genes_important).intersection(set(list_genes_true))
    precision = len(set_intersection)/len(list_genes_important)

    return precision

def recall_genes_score(list_genes_important, list_genes_true):
    set_intersection = set(list_genes_important).intersection(set(list_genes_true))
    recall = len(set_intersection)/len(list_genes_true)

    return recall

def f1_genes_score(precision, recall):
    return 2*precision*recall/(precision+recall)

    
