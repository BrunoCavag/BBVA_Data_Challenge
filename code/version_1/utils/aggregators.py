import pandas as pd
import numpy as np

def media_0(col:pd.core.series.Series):
    return 0 if col.sum() == 0 else np.mean(col[col>0.0])

def var_0(col:pd.core.series.Series):
    return 0 if col.sum() == 0 else np.var(col[col>0.0])

def iqr_range(col:pd.core.series.Series):
    if col.empty:
        return 0
    q25 = np.quantile(col.to_numpy(),0.25)
    q75 = np.quantile(col.to_numpy(),0.75)
    return q75-q25

def count_anom_low(col:pd.core.series.Series):
    if col.empty:
        return 0
    iqr = iqr_range(col)
    lower = np.quantile(col.to_numpy(),0.25) - 1.5*(iqr)
    return (col.to_numpy() < lower).sum()

def count_anom_top(col:pd.core.series.Series):
    if col.empty:
        return 0
    iqr = iqr_range(col)
    top = np.quantile(col.to_numpy(),0.75) + 1.5*(iqr)
    return (col.to_numpy() > top).sum()