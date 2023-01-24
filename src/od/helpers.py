import os
os.chdir(os.environ['PROJECT_DIR'])

import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import numpy as np

from pyod.models.copod import COPOD
from pyod.models.abod import ABOD
from pyod.models.lof import LOF
import sklearn.metrics.pairwise as pairwise
from sklearn.neighbors import LocalOutlierFactor

import umap.umap_ as umap
from sklearn.decomposition import PCA

# visualization
def _display_outliers(all_scores):
    subplots_adjust(hspace=0.000)
    n_subplots = len(all_scores)

    # dynamic amount of plots using 1 column
    for i in range(n_subplots):
        ax = subplot(n_subplots, 1, i+1)
        ax.plot(range(len(all_scores[i])), all_scores[i])

# outlier detection methods
def copod(X):    
    detector = COPOD()
    scores = detector.decision_function(X)
    return scores

def abod(X):
    detector = ABOD().fit(X)
    scores = detector.decision_scores_
    return scores

def csod(X):
    '''
    Custom cosine similarity based outlier detection

    Steps:

    1. Compute Pairwise cosine similarities of all vectors
    2. Fill diagnol of matrix with zeros (so they aren't included in the final calculation)
    3. Compute mean of each row - each represents the similarity of vector n with others

    Returns
    ------
    np.array
        similarity scores - lower the score, higher the outlier
    '''
    cosine_matrix = pairwise.cosine_similarity(X, X)
    np.fill_diagonal(cosine_matrix, 0)
    scores = np.mean(cosine_matrix, axis=1)

    return scores

def lof(X, n_neighbors=20):
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1).fit(X)
    scores = clf.negative_outlier_factor_

    return scores

# dimensionality reduction methods
def pca_reduce(X, dimension):
    pca = PCA(n_components=dimension)
    return pca.fit_transform(X)
    
def umap_reduce(X, dimension):
    _umap = umap.UMAP(random_state=1, n_components=dimension)
    return _umap.fit_transform(X)

def tsne_reduce(X, dimension):
    return None
    
# handle different kinds of feature vectors (bert, tfidf, etc.)
def reduce_vectors(feature_vectors, reduce_method, dimension):
    '''
    Parameters
    ----------
    feature_vectors : np.array
        vector of vectors with different types of document embeddings e.g, [bert_embeddings, tfidf_embeddings, ...]
    reduce_method : function
        python function for dimensionality reduction on each set of feature vectors
    dimension: int
        dimension to reduce vectors

    Returns
    -------
    array
        array of vectors representing each set of reduced feature vectors
    '''
    reduced_vectors = [reduce_method(vectors, dimension) for vectors in feature_vectors]

    return reduced_vectors

def test_outliers(feature_vectors, od_method):
    '''
    Parameters
    ----------
    feature_vectors : np.array
        vector of vectors with different types of document embeddings e.g, [bert_embeddings, tfidf_embeddings, ...>
    od_method : function
        python function for outlier detection on each set of feature vectors

    Returns
    -------
    array
        array of vectors with scores from each outlier detection method
    '''
    all_scores = [od_method(vectors) for vectors in feature_vectors]
    _display_outliers(all_scores)

    return all_scores