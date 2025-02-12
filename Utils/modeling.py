from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from umap import UMAP
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, silhouette_score

print("modeling.py loaded successfully")

def clustering_plot(X, y, labels, title):
    """
    Plot the clustering results
    Inputs:
    - X: DataFrame containing the data
    - y: DataFrame containing the true labels
    - labels: DataFrame containing the predicted labels
    - title: Title of the plot
    Returns:
    - None
    """
    reduced_X = pd.DataFrame(UMAP(n_components=2).fit_transform(X), columns=['X1', 'X2'])
    reduced_X['cluster'] = labels
    if accuracy_score(y, labels) < 0.5:
        reduced_X['cluster'] = 1 - reduced_X['cluster']
    reduced_X['y'] = y
    sns.scatterplot(x='X1', y='X2', hue='y', data=reduced_X, palette='viridis')
    sns.kdeplot(x='X1', y='X2', hue='cluster', data=reduced_X, palette='viridis')
    plt.title(title)
    plt.grid(True)
    plt.show()
    accuracy = accuracy_score(y, reduced_X['cluster'])
    print(f"Silhouette Score: {silhouette_score(X, labels):.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    return accuracy


def dbscan_labels(X):
    """
    Compute the DBSCAN labels, ensuring that there are at least two clusters
    Inputs:
    - X: DataFrame containing the data
    Returns:
    - labels: DataFrame containing the predicted labels
    """
    count = 0
    while count < 1000000:
        count += 100
        dbscan = DBSCAN(eps=count, min_samples=5).fit(X)
        if len(set(dbscan.labels_)) > 1: break
    return dbscan.labels_ + 1


def logistic_regression(X_tr, y_tr, X_ts):
    """
    Perform logistic regression
    Inputs:
    - X_tr: DataFrame containing the training data
    - y_tr: DataFrame containing the training labels
    - X_ts: DataFrame containing the test data
    Returns:
    - y_pred: DataFrame containing the predicted labels
    """
    lr = LogisticRegression(random_state=1)
    lr.fit(X_tr, y_tr)
    y_pred = lr.predict(X_ts)
    return y_pred


def svm(X_tr, y_tr, X_ts, kernel='linear'):
    """
    Perform SVM
    Inputs:
    - X_tr: DataFrame containing the training data
    - y_tr: DataFrame containing the training labels
    - X_ts: DataFrame containing the test data
    - kernel: Kernel type
    Returns:
    - y_pred: DataFrame containing the predicted labels
    - svm: SVM model
    """
    svm = SVC(kernel=kernel)
    svm.fit(X_tr, y_tr)
    y_pred = svm.predict(X_ts)
    return y_pred, svm


def plot_decision_boundary(X, y, model, title):
    """
    Plot the decision boundary of the 2D model
    Inputs:
    - X: DataFrame containing the data
    - y: DataFrame containing the labels
    - model: Model
    - title: Title of the plot
    Returns:
    - None
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

    
def random_forest(X_tr, y_tr, X_ts, n_estimators=100):
    """
    Perform random forest
    Inputs:
    - X_tr: DataFrame containing the training data
    - y_tr: DataFrame containing the training labels
    - X_ts: DataFrame containing the test data
    - n_estimators: Number of trees in the forest
    Returns:
    - y_pred: DataFrame containing the predicted labels
    """
    predictions = []
    for i in range(n_estimators):
        rf = DecisionTreeClassifier()
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_ts)
        predictions.append(y_pred)
    y_pred = np.mean(predictions, axis=0)
    return y_pred>0.5


def neural_network(X_tr, y_tr, X_ts):
    """
    Perform neural network
    Inputs:
    - X_tr: DataFrame containing the training data
    - y_tr: DataFrame containing the training labels
    - X_ts: DataFrame containing the test data
    Returns:
    - y_pred: DataFrame containing the predicted labels
    """
    nn = MLPClassifier(random_state=1)
    nn.fit(X_tr, y_tr)
    y_pred = nn.predict(X_ts)
    return y_pred


def plot_models(spca_HCC_score, stsne_HCC_score, sumap_HCC_score, kmeans_HCC_score, gmm_HCC_score, dbscan_HCC_score, spectral_HCC_score, lr_HCC_score, svm_HCC_lin_score, svm_HCC_rbf_score, rf_HCC_score, nn_HCC_score, spca_MCF_score, stsne_MCF_score, sumap_MCF_score, kmeans_MCF_score, gmm_MCF_score, dbscan_MCF_score, spectral_MCF_score, lr_MCF_score, svm_MCF_lin_score, svm_MCF_rbf_score, rf_MCF_score, nn_MCF_score):
    """
    Plot the model accuracies
    Inputs:
    - accuracy scores for each model for both HCC1806 and MCF7
    Returns:
    - None
    """
    plt.figure(figsize=(20,8))
    plt.subplot(1, 2, 1)
    models = ['PCA', 't-SNE', 'UMAP','KMeans', 'GMM', 'DBSCAN', 'Spectral Clustering', 'Logistic Regression', 'SVM (Linear)', 'SVM (RBF)', 'Random Forest', 'Neural Network']

    HCC_scores = [spca_HCC_score, stsne_HCC_score, sumap_HCC_score, kmeans_HCC_score, gmm_HCC_score, dbscan_HCC_score, spectral_HCC_score, lr_HCC_score, svm_HCC_lin_score, svm_HCC_rbf_score, rf_HCC_score, nn_HCC_score]
    plt.bar(models, HCC_scores, color='blue', alpha=0.7)
    plt.title('HCC1806: Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=55)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    MCF_scores = [spca_MCF_score, stsne_MCF_score, sumap_MCF_score, kmeans_MCF_score, gmm_MCF_score, dbscan_MCF_score, spectral_MCF_score, lr_MCF_score, svm_MCF_lin_score, svm_MCF_rbf_score, rf_MCF_score, nn_MCF_score]
    plt.bar(models, MCF_scores, color='red', alpha=0.7)
    plt.title('MCF7: Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=55)
    plt.grid(True)
    plt.show()