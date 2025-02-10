from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from openTSNE import TSNE
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from umap import UMAP

print('dim_reduction.py loaded')

class SPCA(PCA):
    def __init__(self, n_components=2, k=5):
        super().__init__(n_components=n_components)
        self.k = k

    def fit(self, X, y):
        self.y = y
        self.kbest = SelectKBest(score_func=f_classif, k=self.k).fit(X, y)
        X_selected = self.kbest.transform(X)
        super().fit(X_selected)

    def transform(self, X):
        X_selected = self.kbest.transform(X)
        return super().transform(X_selected)

 
class STSNE(TSNE):
    def __init__(self, n_components=2):
        super().__init__(n_components=n_components)

    def fit(self, X, y):
        self.nca = NeighborhoodComponentsAnalysis(n_components=self.n_components)
        self.nca = self.nca.fit(X, y)
        X_nca = self.nca.transform(X)
        self.model = super().fit(X_nca)

    def transform(self, X):
        X_nca = self.nca.transform(X)
        return self.model.transform(X_nca)


def plot_pca_ev(df, title):
    pca = PCA()
    pca.fit(df)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title(f'{title}: Explained Variance Ratio with PCA')
    plt.grid(True)
    plt.show()


def plot_2d(df, y, data_name, dim_red_name):
    df['y'] = y
    sns.scatterplot(x='X1', y='X2', hue='y', data=df, palette='viridis', alpha=0.5)
    sns.kdeplot(x='X1', y='X2', hue='y', data=df, palette='viridis')
    plt.title(f'{data_name}: 2D {dim_red_name} divided into hypoxia and normoxia')
    plt.grid(True)
    plt.show()


def supervised_pca(X, y, k=5):
    spca = SPCA(k=k)
    spca.fit(X, y)
    X_spca = pd.DataFrame(spca.transform(X), columns=['X1', 'X2'])
    return X_spca, spca

def supervised_tsne(X, y):
    stsne = STSNE()
    stsne.fit(X, y)
    X_stsne = pd.DataFrame(stsne.transform(X), columns=['X1', 'X2'])
    return X_stsne, stsne

def supervised_umap(X, y):
    umap = UMAP(n_components=2)    
    umap.fit(X, y)
    X_umap = pd.DataFrame(umap.transform(X), columns=['X1', 'X2'])
    return X_umap, umap

def dim_red_predictor(X_reduced_tr, y_tr, X_ts, model):
    X_reduced_ts = model.transform(X_ts)
    y_ts = KNeighborsClassifier().fit(X_reduced_tr, y_tr).predict(X_reduced_ts)
    return y_ts
