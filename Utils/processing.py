import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

print("processing.py loaded")


def get_oxia(df):
    cells = df.index
    return [1 if "Hypo" in cell else 0 for cell in cells]


def split_hypo_norm(df):
    df["y"] = get_oxia(df)
    df_hypo = df[df["y"] == 1]
    df_norm = df[df["y"] == 0]
    df_hypo.drop(columns=["y"], inplace=True)
    df_norm.drop(columns=["y"], inplace=True)
    return df_hypo, df_norm


def correlation_heatmap(df_HCC, df_MCF):
    # Identify genes with same expression in correlation heatmap
    plt.figure(figsize=(20,8))
    plt.subplot(1, 2, 1)
    sns.heatmap(df_HCC.corr(), cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title('HCC1806')
    plt.subplot(1, 2, 2)
    sns.heatmap(df_MCF.corr(), cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title('MCF7')
    plt.suptitle('Correlation heatmap of unfiltered data')
    plt.show()


def rescale_normalize(df):
    scaler = StandardScaler()
    df_rescaled = scaler.fit_transform(df)
    return pd.DataFrame(df_rescaled, index=df.index, columns=df.columns)


def smallest_correlation(df, n=10):
    return list(df.corr().mean().nsmallest(n).index.values)

def largest_correlation(df, n=10):
    return list(df.corr().mean().nlargest(n).index.values)


def frac_zeros(df):
    return round((((df == 0).stack().sum())/(df.shape[0] * df.shape[1])) * 100, 2)