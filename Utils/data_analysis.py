import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Utils.processing import *

print("data_analysis.py loaded")

def plot_gene_expression(df_HCC, df_MCF):
    plt.figure(figsize=(20,8))
    plt.subplot(1, 2, 1)
    plt.hist(df_HCC.sum(axis=1), bins=100, color='blue', edgecolor='black', alpha=0.7, log=True)
    plt.title('HCC1806: Number of cells per gene')
    plt.xlabel('Number of cells')
    plt.ylabel('Number of genes')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.hist(df_MCF.sum(axis=1), bins=100, color='red', edgecolor='black', alpha=0.7, log=True)
    plt.title('MCF7: Number of cells per gene')
    plt.xlabel('Number of cells')
    plt.ylabel('Number of genes')
    plt.grid(True)
    plt.show()


def plot_zeros(df_HCC, df_MCF):
    plt.figure(figsize=(20,8))
    plt.subplot(1, 2, 1)
    plt.hist((df_HCC == 0).sum(axis=1), bins=100, color='blue', edgecolor='black', alpha=0.7, log=True)
    plt.title('HCC1806: Number of zeros per gene')
    plt.xlabel('Number of zeros')
    plt.ylabel('Number of genes')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.hist((df_MCF == 0).sum(axis=1), bins=100, color='red', edgecolor='black', alpha=0.7, log=True)
    plt.title('MCF7: Number of zeros per gene')
    plt.xlabel('Number of zeros')
    plt.ylabel('Number of genes')
    plt.grid(True)
    plt.show()


def plot_top_genes(df_HCC, df_MCF):
    # Extract most expressed genes
    top_genes_HCC = df_HCC.sum(axis=1).sort_values(ascending=False).head(10).index
    top_genes_MCF = df_MCF.sum(axis=1).sort_values(ascending=False).head(10).index

    # Plot the distribution of the most expressed genes
    plt.figure(figsize=(20,8))
    plt.subplot(1, 2, 1)
    for gene in top_genes_HCC:
        sns.kdeplot(df_HCC.loc[gene], label=gene)
    plt.title('HCC1806: Distribution of the most expressed genes')
    plt.xlabel('Gene expression')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for gene in top_genes_MCF:
        sns.kdeplot(df_MCF.loc[gene], label=gene)
    plt.title('MCF7: Distribution of the most expressed genes')
    plt.xlabel('Gene expression')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def diff_genes(df_hypo, df_norm):
    diff = []
    for gene in df_hypo.index:
        diff.append((gene, df_hypo.loc[gene].mean() - df_norm.loc[gene].mean()))
    return sorted(diff, key=lambda x: x[1], reverse=True)


def plot_diff_genes(df_HCC, df_MCF):
    # Split the data into hypoxia and normoxia
    df_HCC_hypo, df_HCC_norm = split_hypo_norm(df_HCC.T)
    df_MCF_hypo, df_MCF_norm = split_hypo_norm(df_MCF.T)

    diff_genes_HCC = diff_genes(df_HCC_hypo.T, df_HCC_norm.T)
    diff_genes_MCF = diff_genes(df_MCF_hypo.T, df_MCF_norm.T)

    # Barplot of number of occurences of the top 10 genes with the largest difference in expression between hypoxia and normoxia
    plt.figure(figsize=(20,8))
    plt.subplot(1, 2, 1)
    genes_HCC = [gene[0] for gene in diff_genes_HCC[:10]]
    diffs_HCC = [gene[1] for gene in diff_genes_HCC[:10]]
    sns.barplot(x=genes_HCC, y=diffs_HCC, palette='viridis')
    plt.title('HCC1806')
    plt.xlabel('Gene')
    plt.ylabel('Difference in expression')
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    genes_MCF = [gene[0] for gene in diff_genes_MCF[:10]]
    diffs_MCF = [gene[1] for gene in diff_genes_MCF[:10]]
    sns.barplot(x=genes_MCF, y=diffs_MCF, palette='viridis')
    plt.title('MCF7')
    plt.xlabel('Gene')
    plt.ylabel('Difference in expression')
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.suptitle('Top 10 genes with the largest difference in expression between hypoxia and normoxia')
    plt.show()


def hist_cor(df, title="", k = 3, cells=None):
    if cells is None: 
        c_small = df.corr().sample(n=k,axis='columns')
    else:
        c_small = df.corr().loc[:,cells[:3]]
    sns.histplot(c_small,bins=100)
    plt.title(f"Corellation between {k} cells expression profiles for {title}")
    plt.ylabel('Frequency')
    plt.xlabel('Correlation')
    plt.show()