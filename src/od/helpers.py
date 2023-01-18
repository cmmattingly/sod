import os
os.chdir(os.environ['PROJECT_DIR'])

import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import numpy as np

from pyod.models.copod import COPOD
from pyod.models.abod import ABOD
from pyod.models.lof import LOF
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

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

def ncsod(X):
    # get pairwise cosine similarities
    cosine_matrix = cosine_similarity(X, X)
    # remove same vector pairs in calculation (diagnols=1)
    np.fill_diagonal(cosine_matrix, 0)
    # get the normalized sum of each row
    scores = [sum(row) / len(X) for row in cosine_matrix]

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
        vector of vectors with different types of document embeddings (bert, tfidf, etc.)
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
        vector of vectors with different types of document embeddings (bert, tfidf, etc.)
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