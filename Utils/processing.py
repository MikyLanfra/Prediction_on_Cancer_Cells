import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

print("processing.py loaded")


def get_oxia(df):
    """
    Get the oxia status of each cell
    Inputs:
    - df: DataFrame containing the data
    Returns:
    - List of oxia status for each cell
    """
    cells = df.index
    return [1 if "Hypo" in cell else 0 for cell in cells]


def split_hypo_norm(df):
    """
    Split the data into hypoxic and normoxic
    Inputs:
    - df: DataFrame containing the data
    Returns:
    - df_hypo: DataFrame containing the hypoxic data
    - df_norm: DataFrame containing the normoxic data
    """
    df["y"] = get_oxia(df)
    df_hypo = df[df["y"] == 1]
    df_norm = df[df["y"] == 0]
    df_hypo.drop(columns=["y"], inplace=True)
    df_norm.drop(columns=["y"], inplace=True)
    return df_hypo, df_norm


def correlation_heatmap(df_HCC, df_MCF):
    """
    Plot the correlation heatmap of the data
    Inputs:
    - df_HCC: DataFrame containing the gene expression data for HCC1806
    - df_MCF: DataFrame containing the gene expression data for MCF7
    Returns:
    - None
    """
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
    """
    Rescale and normalize the data
    Inputs:
    - df: DataFrame containing the data
    Returns:
    - df_rescaled: DataFrame containing the rescaled data
    """ 
    scaler = StandardScaler()
    df_rescaled = scaler.fit_transform(df)
    return pd.DataFrame(df_rescaled, index=df.index, columns=df.columns)


def smallest_correlation(df, n=10):
    """
    Get the n smallest correlations
    Inputs:
    - df: DataFrame containing the data
    - n: Number of smallest correlations to return
    Returns:
    - List of n smallest correlations
    """
    return list(df.corr().mean().nsmallest(n).index.values)

def largest_correlation(df, n=10):
    """
    Get the n largest correlations
    Inputs:
    - df: DataFrame containing the data
    - n: Number of largest correlations to return
    Returns:
    - List of n largest correlations
    """
    return list(df.corr().mean().nlargest(n).index.values)


def frac_zeros(df):
    """
    Get the fraction of zeros in the data
    Inputs:
    - df: DataFrame containing the data
    Returns:
    - Fraction of zeros in the data
    """
    return round((((df == 0).stack().sum())/(df.shape[0] * df.shape[1])) * 100, 2)