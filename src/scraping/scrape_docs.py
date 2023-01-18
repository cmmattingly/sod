# Note:
# There are no rules against scraping of the senator website, receiving a 403 error is due to the scripts the government computer system
# employs to monitor network traffic to identify unauthorized attempts to upload or change information, or otherwise cause damage.

# Senate.gov regulations -- https://www.senate.gov/general/privacy.htm:
# Information presented on this site is considered public information and may be distributed or copied unless otherwise specified. 
# Use of appropriate byline/photo/image credits is requested.
# Unauthorized attempts to upload information or change information on this service are strictly prohibited and may be 
# punishable under the Computer Fraud and Abuse Act of 1986 and the National Information Infrastructure Protection Act.


import os
os.chdir(os.environ['PROJECT_DIR'])

import re
import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
import time


# multiprocessing uses pickle, need dill for local serialization -- multiprocess package
from multiprocess import Pool, Value
# scraping
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse


from fake_useragent import UserAgent
from lxml.html import fromstring
from itertools import cycle


# utils
from utils.helpers import handle_get_request, df_to_csv

# google cache url prefix
CACHE_PREFIX = "http://webcache.googleusercontent.com/search?q=cache:"
DOCUMENTS_DIR = "data/raw/sen_statements"

def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def find_missing_ids(true_ids):
    # create lambda to parse id from file name
    parse_id = lambda x: x.split('=')[1].split('.')[0]
    
    # parse ids from documents directory
    found_ids = [parse_id(file_name) for file_name in os.listdir(DOCUMENTS_DIR)]
    
    # use set arithmetic to determine missing ids
    missing_ids = list(set(true_ids) - set(found_ids))
    
    return missing_ids

def scrape_senator(df, member_id):
    urls = df[df.member_id == member_id]['url'].to_list()

    # define local function for multiprocessing
    def scrape_url(url):
        # scrape max 2*n urls per second (n = # of processes)
        time.sleep(1)

        # check if url is valid
        if not validate_url(url):
            return ''

        # use random user agent for each url
        headers = {'User-Agent': UserAgent().random}
        # try cached version first
        try:
            response = requests.get(CACHE_PREFIX + url, headers=headers, timeout=5)
            response.raise_for_status()
        except Exception as e:
            # after cached version fails, try real website
            try:
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status()
            except Exception as e:
                print(e)
                return ''

        # start scraping response html content
        soup = BeautifulSoup(response.content, features='lxml')
        exp = re.compile('<.*?>')

        # create full text from concatenating paragraph elements' text
        text = ''.join([
            para.get_text()
            for para in soup.findAll('p') 
            if (exp.search(str(para)))
        ])

        return text
    
    # scrape urls using multiprocessing
    with Pool(processes=2) as pool:
        texts = pool.map(scrape_url, urls)

    return urls, texts

def main():
    # get scraped url data
    sen_urls = pd.read_csv("data/raw/sen_urls.csv")
    # get scraped member ids
    member_ids = sen_urls['member_id'].unique()

    # find ids that we havent scraped yet
    missing_ids = find_missing_ids(member_ids)

    # [TODO-1]: Base missing ids on subfolders of date scraped

    # scrape ids that are missing
    for i, member_id in enumerate(tqdm(missing_ids)):
        urls, texts = scrape_senator(sen_urls, member_id)

        # create and save dataframe using scraped data and urls scraped
        df = pd.DataFrame(list(zip(urls, texts)), columns=['url', 'text'])
        df = df.dropna()
        df_to_csv(df, path=f"data/raw/documents/senator_statements_id={member_id}.csv", verbose=True)

        if not i == 0 and not (i % 5):
            print("Sleeping...")
            time.sleep(5 * 60)
            print("Resuming...")

if __name__ == "__main__":
    main()