import os
os.chdir(os.environ['PROJECT_DIRECTORY'])

import pandas as pd
import glob

from utils.helpers import df_to_csv

DATA_DIR = "data/raw/sen_statements"

def main():
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

    df_to_csv(final_df, "data/processed/sen_statements.csv", verbose=True)

if __name__ == '__main__':
    main()