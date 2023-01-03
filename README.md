# Senator Outlier Detection

Description of project

## Prerequisites
This project requires Python 3 and pip to install required packages using requirements.txt.

You will also need to have software installed to run and execute a Jupyter Notebook.

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
│   ├── interim
│   ├── processed
│   │   └── sen_statements.csv
│   ├── raw
│   │   ├── sen_metadata.csv
│   │   ├── sen_statements
│   │   │   └── senator_statements_id=*.csv
│   │   └── sen_urls.csv
│   ├── utils
│   │   └── state_abbreviations.txt
│   └── vectorized
├── notebooks
│   └── 1.0-outlier-detection-cmmattingly.ipynb
├── setup.sh
└── src
    ├── __init__.py
    ├── data
    │   ├── get_doc_urls.py
    │   ├── get_sen_metadata.py
    │   └── merge_sen_statements.py
    ├── db
    │   └── fetch_docs.py
    ├── feature_extraction
    │   ├── extract_features.py
    │   └── temp_feature_extraction.ipynb
    ├── models
    │   ├── Doc2VecTransformer.py
    │   ├── LDAMallet.py
    │   ├── __init__.py
    │   └── __pycache__
    │       ├── Doc2VecTransformer.cpython-39.pyc
    │       ├── LDAMallet.cpython-39.pyc
    │       └── __init__.cpython-39.pyc
    ├── od
    │   ├── __init__.py
    │   └── copod.py
    ├── scraping
    │   ├── scrape_docs.py
    │   └── scrape_sen_metadata.py
    └── utils
        ├── __init__.py
        ├── __pycache__
        │   ├── __init__.cpython-39.pyc
        │   └── helpers.cpython-39.pyc
        └── helpers.py
```

## Code
The feature extraction script should be the only script needed to get started running the outlier detection jupyer notebook.

### Feature Extraction
Run the following script to extract features from senator statments already created within this github repo:
```bash
python3 src/data/extract_features.
```

### Notebook
Run the following command to perform analysis (outlier detection) on data:
```bash
jupyer notebook notebooks/{version_number}-outlier-detection.ipynb
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
The src/scraping directory contains scripts to both scrape senator metadata (name, party, state, etc.) and senator statments. The scrape_sen_metadata.py script should not be used unless the ProPublica API is down.

Run the following command to scrape senator statements (if you need to update statements):
```bash
python3 src/scraping/scrape_docs.py
```
This python file with create a csv file for each senator containing there latest press releates (max 100). These files can be found in the data/raw/sen_statements directory.
