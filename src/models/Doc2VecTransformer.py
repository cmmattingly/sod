import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

from sklearn.base import BaseEstimator
from sklearn import utils as skl_utils
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

# custom BaseEstimator for training Doc2Vec model and transforming text data
class Doc2VecTransformer(BaseEstimator):
    def __init__(self, vector_size=100, learning_rate=0.01, epochs=20):
        self.lr = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        self.workers = mp.cpu_count() - 1

    def fit(self, X):
        # tag documents for Doc2Vec training - https://radimrehurek.com/gensim/models/doc2vec.html
        tagged_X = [
            TaggedDocument(word_tokenize(doc), [i]) 
            for i, doc in enumerate(X)
        ]
        
        # initalize doc2vec model
        model = Doc2Vec(documents=tagged_X, vector_size=self.vector_size, workers=self.workers)
        
        # pass 1 epoch at a time for TQDM progress 
        for epoch in tqdm(range(self.epochs)):
            model.train(skl_utils.shuffle([x for x in tagged_X]), total_examples=len(tagged_X), epochs=1)
            model.alpha -= self.lr
            model.min_alpha = model.alpha
        
        # store model for transforming
        self._model = model
        return self

    def transform(self, X):
        return np.array([
            self._model.infer_vector(word_tokenize(doc))
            for doc in X
        ])