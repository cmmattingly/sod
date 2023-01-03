import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

from sklearn.base import BaseEstimator
from sklearn import utils as skl_utils
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

class Doc2VecTransformer(BaseEstimator):
    def __init__(self, vector_size=100, learning_rate=0.01, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        self.workers = mp.cpu_count() - 1

    def fit(self, X):
        tagged_X = [
            TaggedDocument(doc.split(), [i]) 
            for i, doc in enumerate(X)
        ]
        
        model = Doc2Vec(documents=tagged_X, vector_size=self.vector_size, workers=self.workers)

        for epoch in tqdm(range(self.epochs)):
            model.train(skl_utils.shuffle([x for x in tagged_X]), total_examples=len(tagged_X), epochs=1)
            model.alpha -= self.learning_rate
            model.min_alpha = model.alpha

        self._model = model
        return self

    def transform(self, X):
        return np.asmatrix(np.array([
            self._model.infer_vector(doc.split())
            for doc in X
        ]))