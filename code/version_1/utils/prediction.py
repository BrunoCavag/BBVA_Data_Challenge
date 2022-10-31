import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


def prediccion(df:pd.DataFrame, category_column:str, target_column:str, pred_column):
    cat_counts = []
    f1_scores = []
    categories = df[category_column].unique()
    for cat in categories:
        temp = df[df[category_column]==cat]
        cat_counts.append(len(temp)/len(df))
        f1_scores.append(f1_score(temp[target_column],temp[pred_column]))

    return np.sum(np.array(cat_counts) * np.array(f1_scores))