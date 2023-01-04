import os
os.chdir(os.environ['PROJECT_DIRECTORY'])

import pandas as pd
import glob
from tqdm import tqdm

import gensim.parsing.preprocessing as gsp
from gensim import utils

import nltk
from nltk import tokenize
nltk.download('stopwords')

from utils.helpers import df_to_csv

DATA_DIR = "data/raw/sen_statements"
FILTERS = [gsp.strip_tags, 
           gsp.strip_punctuation, 
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric, 
           gsp.remove_stopwords, 
           gsp.strip_short]
STOP_WORDS = ['sen', 'senator', 'senate', 'senators']

def clean_text(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in FILTERS:
        s = f(s)
    return s

def process_text(text):
    # clean unprocessed data using clean_text function
    try:
        text = clean_text(text)
    except:
        print(text)
    
    # word tokenize data entry -> remove tokens in stop words and that are one character long
    text_tokens = tokenize.word_tokenize(text)
    tokens = [
        word 
        for word in text_tokens 
        if not word in STOP_WORDS and len(word) > 1
    ]
    text = " ".join(tokens)
    
    return text

# [TODO-3] Remove when using fetched statements
def merge_sen_statements():
    # get senator statements meta data
    sen_urls = pd.read_csv("data/raw/sen_urls.csv")

    # [TODO-1]: Take into account date scraped when making corpus

    # retrieve senator statement paths
    file_names = [
        file_name
        for file_name in glob.glob(f"{DATA_DIR}/*.csv")
    ]

    # concat csv files together
    sen_statements = pd.concat([
        pd.read_csv(file_name)
        for file_name in file_names
    ])
    
    # merge dataframes using url column
    final_df = pd.merge(sen_urls, sen_statements, on='url')

    df_to_csv(final_df, "data/raw/sen_statements.csv", verbose=True) 

    return final_df


def main():
    # [TODO-3] Process fetched statements instead of local ones 
    try:
        sen_statements = pd.read_csv("data/raw/sen_statements.csv")
    except Exception as e:
        sen_statements = merge_sen_statements()

    # [TODO-3] REMOVE drop na 
    sen_statements = sen_statements.dropna()

    # use tqdm pandas for progress apply
    tqdm.pandas()
    
    print('Processing Text...')
    process_lambda = lambda row: process_text(row['text'])
    sen_statements['text'] = sen_statements.progress_apply(process_lambda, axis=1)

    df_to_csv(sen_statements, path="data/processed/sen_statements.csv", verbose=True)

if __name__ == "__main__":
    main()