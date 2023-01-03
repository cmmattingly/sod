import requests
import pandas as pd

#scraping
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

# helper functions

# api calls
def handle_get_request(url, headers, verbose=False):
    if verbose:
        print(f"Get request at {url}")

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(e, url)
        return None

    return response

def handle_post_request(url, headers, data, verbose=False):
    if verbose:
        print(f"Post request at {url}")

    try:
        response = requests.post(url, data=data, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(e, url)
        return None

    return response

# persistence
def df_to_csv(df, path, verbose=False):
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        print(e)
    
    if verbose:
        print(f"Converted dataframe to csv at {path}")