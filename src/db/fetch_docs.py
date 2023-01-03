import os
os.chdir(os.environ['PROJECT_DIRECTORY'])

import pandas as pd
import json
import requests
from tqdm import tqdm 
import argparse

import gensim.parsing.preprocessing as gsp
from gensim import utils

from helpers import handle_get_request, df_to_csv

import nltk
from nltk import tokenize
nltk.download('stopwords')

# define additional stop words and filters for text
FILTERS = [gsp.strip_tags, 
           gsp.strip_punctuation, 
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric, 
           gsp.remove_stopwords, 
           gsp.strip_short]

STOP_WORDS = ['sen', 'senator', 'senate', 'senators']

# helper functions
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    return args

def clean_text(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in FILTERS:
        s = f(s)
    return s

def process_text(text):
    # clean unprocessed data using clean_text function
    text = clean_text(text)
    
    # word tokenize data entry -> remove tokens in stop words and that are one character long
    text_tokens = tokenize.word_tokenize(text)
    tokens = [word for word in text_tokens if not word in STOP_WORDS and len(word) > 1]
    text = " ".join(tokens)
    
    return text

def process_response(response):
    print("Processing Response", end=" -- ")

    # convert request text to dictionary
    json_data = json.loads(response.text)
    
    # obtain elements of each json object and parse into 2d array
    parsed_data = [
        [o['doc_id'], o['sen_id'], o['doc_title'], process_text(o['doc_text']), o['doc_link']] 
        for o in json_data
    ]
    
    # convert 2d array into dataframe
    df = pd.DataFrame(parsed_data, columns=['doc_id', 'sen_id', 'doc_title', 'doc_text', 'doc_link'])
    
    print("Success")
    return df


def main():
    # parse args
    args = parse_args()
    path = args.path

    # fetch from docs endpoint and convert to json
    endpoint = "http://das-lab.org:1702/docs"
    
    # get request and process response
    res = handle_get_request(endpoint, verbose=True)
    df = process_response(res)

    # convert dataframe to csv
    df_to_csv(df, path, verbose=True)


if __name__ == "__main__":
    main()