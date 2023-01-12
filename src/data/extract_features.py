import os
os.chdir(os.environ['PROJECT_DIR'])

import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import argparse

from sentence_transformers import SentenceTransformer

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', nargs='?', const=True, type=bool, default=False)
    parser.add_argument('--id', nargs='?', type=str, default='ALL')
    args = parser.parse_args()

    return args

def tfidf_extract(docs):
    print("Extracting TF-IDF Features...")

    vectorizer = TfidfVectorizer()
    doc_embeddings = vectorizer.fit_transform(docs)

    return doc_embeddings.toarray()

def doc2vec_extract(docs):
    print("Extracting Doc2Vec Features...")

    model = Doc2VecTransformer().fit(docs)
    doc_embeddings = model.transform(docs)

    return doc_embeddings

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

def bert_extract(docs):
    # https://huggingface.co/docs/transformers/model_doc/bert
    # Use transformers instead of sentence transformers
    print("Extracting BERT Features...")

    sbert_model_distilro = SentenceTransformer('all-distilroberta-v1')
    doc_embeddings = sbert_model_distilro.encode(docs)

    return doc_embeddings

def main():
    args = parse_args()
    test = args.test
    member_id = args.id

    # get processed data (either test or senator statements) + define path for saving embeddings
    if test:
        df = pd.read_csv("data/processed/test_dataset.csv")
        path_prefix = "data/vectorized/test_dataset/" 
    else:
        df = pd.read_csv("data/processed/sen_statements.csv")
        path_prefix = f"data/vectorized/sen_statements/id={member_id}/"  
    
    try:
        os.mkdir(path_prefix)
    except OSError as e:
        print(f"{path_prefix} already exists. Will overwrite existing vectorized statements.")

    # if member id is default ('ALL') use all rows -- else use specific member id
    processed_statements = (
        df['text'].values.astype('U') 
        if member_id == 'ALL'
        else df[df.member_id == member_id]['text'].values.astype('U'))

    # loop through different feature extractors
    feature_extractors = [tfidf_extract, doc2vec_extract, bert_extract]
    for feature_extractor in feature_extractors:
        # extract features using current custom function 
        doc_embeddings = feature_extractor(processed_statements)

        # save embeddings using feature extractor function name
        np.save(path_prefix + f"{feature_extractor.__name__.split('_')[0]}_vectors", doc_embeddings)

if __name__ == "__main__":
    main()