import os
os.chdir(os.environ['PROJECT_DIRECTORY'])

import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn import utils as skl_utils

import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

from nltk.tokenize import regexp
from nltk import tokenize

# import custom models
from models.LDAMallet import *
from models.Doc2VecTransformer import Doc2VecTransformer

MAX_TOPICS = 15

def tfidf_extract(docs):
    print("Extracting TF-IDF Features...")

    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(docs)

    return feature_vectors.toarray()

def doc2vec_extract(docs):
    print("Extracting Doc2Vec Features...")

    model = Doc2VecTransformer().fit(docs)
    feature_vectors = model.transform(docs)

    return feature_vectors

def lda_extract(docs):
    print("Extracting LDA Features...")

    # lda pre-processing
    tokenizer = regexp.RegexpTokenizer(r"\w+")
    texts = list(map(lambda x: tuple(tokenizer.tokenize(x)), docs))
    id2word = corpora.Dictionary(texts)
    corpus = list(map(lambda x: id2word.doc2bow(x), texts))

    # define function for coherence optimization (find best number of topics)
    def coherence_optimization(dictionary, corpus, texts, limit=MAX_TOPICS, start=5, step=5):
        model_list, coherence_values = [], []
        for num_topics in tqdm(range(start, limit, step)):
            model = LdaMallet(os.environ['MALLET_DIR'], corpus=corpus, num_topics=num_topics, id2word=id2word)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_npmi')
            
            model_list.append(model)
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values

    # perform coherence optimization
    model_list, coherence_values = coherence_optimization(dictionary=id2word, corpus=corpus, texts=texts)
    
    # get best model based on best coherence 
    model = model_list[np.argmax(coherence_values)]
    # create document topic matrix
    doc_top_matrix = np.array([*model.load_document_topics()])

    return doc_top_matrix

def main():
    # get processed senator statements
    # [TODO-3] Change statements to fetched statements for senator analysis

    sen_statements = pd.read_csv("data/processed/sen_statements.csv")
    processed_statements = sen_statements['text'].values.astype('U')

    feature_extractors = [tfidf_extract, doc2vec_extract, lda_extract]

    for feature_extractor in feature_extractors:
        vectorized = feature_extractor(processed_statements)
        np.save(f"data/vectorized/{feature_extractor.__name__.split('_')[0]}_vectors", vectorized)

if __name__ == "__main__":
    main()