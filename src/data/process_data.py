import os
os.chdir(os.environ['PROJECT_DIR'])

import pandas as pd
import glob
from tqdm import tqdm
tqdm.pandas()

import argparse

import gensim.parsing.preprocessing as gsp
from gensim import utils

import nltk
from nltk import tokenize

try:
    nltk.corpus.stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

from utils.helpers import df_to_csv

# define text filters for preprocessing
FILTERS = [
    gsp.strip_tags,
    gsp.strip_punctuation, 
    gsp.strip_multiple_whitespaces,
    gsp.strip_numeric,
    gsp.strip_short
]

# use nltk stopwords and add custom stop words
CUSTOM_STOP_WORDS = ['senator', 'senate', 'senators']
STOP_WORDS = nltk.corpus.stopwords.words('english').extend(CUSTOM_STOP_WORDS)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', nargs='?', const=True, type=bool, default=False)
    args = parser.parse_args()

    return args

def process_text(text):
    # lowercase text and convert to unicode
    text = utils.to_unicode(x.lower(text))

    # remove stop words first due to stemmed words not being included in nltk stop words
    tokens = tokenize.word_tokenize(text)
    processed_tokens = [
        word 
        for word in tokens
        if not word in STOP_WORDS
    ]
    text = " ".join(processed_tokens)

    # run text through gensim filters
    text = gsp.preprocess_string(text, FILTERS)
    
    return text

# [TODO-3] Remove when using fetched statements
def merge_sen_statements():
    # [TODO-1]: Take into account date scraped when making corpus
    # merge csv files together using filenames from glob.glob
    file_names = [file_name for file_name in glob.glob(f"data/raw/sen_statements/*.csv")]
    sen_statements = pd.concat([pd.read_csv(file_name) for file_name in file_names])

    # get senator statement url meta data
    sen_urls = pd.read_csv("data/raw/sen_urls.csv")

    # merge url metadata and text using url column
    final_df = pd.merge(sen_urls, sen_statements, on='url')
    
    df_to_csv(final_df, "data/raw/sen_statements.csv", verbose=True) 

    return final_df

def main():
    # get test argument for determining what dataset to process
    args = parse_args()
    test = args.test

    # [TODO-3] Process fetched statements instead of local ones     
    if test:
        df = pd.read_csv("data/utils/test_dataset.csv")
    else:
        try:
            df = pd.read_csv("data/raw/sen_statements.csv")
        except Exception as e:
            df = merge_sen_statements()

    # [TODO-3] REMOVE drop na (no documents should be NA from database)
    df = df.dropna()

    # use tqdm pandas progress apply
    print('Processing Text...')
    process_lambda = lambda row: process_text(row['text'])
    df['text'] = df.progress_apply(process_lambda, axis=1)

    # drop NA due to possibly empty documents after processing (only stop words, etc.)
    df = df.dropna()

    csv_name = "test_dataset" if test else "sen_statements"
    df_to_csv(df, path=f"data/processed/{csv_name}.csv", verbose=True)

if __name__ == "__main__":
    main()