# Senator Outlier Detection

Description of project

## Prerequisites
1. python3
2. jupyter notbeook
3. Mallet on machine: https://mallet.cs.umass.edu/download.php

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/username/project-name.git
```

2. Navigate to project directory:
```bash
cd senator_outlier_detection
```

3. Create and activate virtual environment:
```bash
virtualenv env
source env/bin/activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

5. (IMPORTANT) Setup environment variables (PYTHONPATH, API KEY, etc.) for project:
```bash
source setup.sh
```

## Project Tree
```bash
.
├── README.md
├── data
│   ├── fetched
│   │   └── docs.csv
│   ├── processed
│   │   ├── sen_statements.csv
│   │   └── test_dataset.csv
│   ├── raw
│   │   ├── sen_metadata.csv
│   │   ├── sen_statements
│   │   │   └── senator_statements_id=*.csv
│   │   ├── sen_statements.csv
│   │   └── sen_urls.csv
│   ├── utils
│   │   ├── sen-id_to_member-id.json
│   │   ├── state_abbreviations.txt
│   │   └── test_dataset.csv
│   └── vectorized
│       ├── sen_statements
│       │   ├── id=*
│       │   │   ├── bert_vectors.npy
│       │   │   ├── doc2vec_vectors.npy
│       │   │   ├── lda_vectors.npy
│       │   │   └── tfidf_vectors.npy
│       └── test_dataset
│           ├── bert_vectors.npy
│           ├── doc2vec_vectors.npy
│           └── tfidf_vectors.npy
├── notebooks
│   ├── 1.0-outlier-detection-exploration.ipynb
│   └── 1.0-sen-outlier-detection.ipynb
├── setup.sh
└── src
    ├── __init__.py
    ├── data
    │   ├── extract_features.py
    │   ├── get_doc_urls.py
    │   ├── get_sen_metadata.py
    │   └── process_data.py
    ├── db
    │   ├── fetch_docs.py
    │   └── push_docs.py
    ├── models
    │   ├── Doc2VecTransformer.py
    │   ├── LDAMallet.py
    │   └── __init__.py
    ├── od
    │   ├── __init__.py
    │   ├── copod.py
    │   └── helpers.py
    ├── scraping
    │   ├── scrape_docs.py
    │   └── scrape_sen_metadata.py
    └── utils
        ├── __init__.py
        └── helpers.py

```

## Code
The feature extraction script should be the only script needed to get started running the outlier detection jupyer notebook.

### Feature Extraction
Run the following script to extract features from senator statments already created within this github repo:
```bash
python3 src/data/extract_features.py
```

### Notebooks
Run the following command to view analysis (outlier detection) on test dataset -- with truth labels:
```bash
jupyter notebook notebooks/{version_number}-outlier-detection-exploration.ipynb
```

Run the following command to view analysis on senator statements dataset -- no truth labels:
Note: This repo may be updated with truth label data once the project is taken over (content analysis team).
```bash
jupyter notebook notebooks/{version_number}-sen-outlier-detection.ipynb
```

### ProPublica API
To access the ProPublica api to obtain the needed data for this project (if you are running from scratch), use the get_*.py scripts within the src/data directory. For example:

1. Obtain required metadata (member_id, name, party, state, etc):
```bash
python3 src/data/get_sen_metadata.py
```

2. Using the acquired metadata, obtain senator statment urls for scraping:
```bash
python3 src/data/get_doc_urls.py
```

### Scraping
The src/scraping directory contains scripts to both scrape senator metadata (name, party, state, etc.) and senator statements. The scrape_sen_metadata.py script should not be used unless the ProPublica API is down.

Run the following command to scrape senator statements (if you need to update statements):
```bash
python3 src/scraping/scrape_docs.py
```
This python file with create a csv file for each senator containing there latest press releates (max 100). These files can be found in the data/raw/sen_statements directory.
