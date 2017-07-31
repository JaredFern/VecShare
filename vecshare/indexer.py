from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium import webdriver
from bs4 import BeautifulSoup
import datadotworld as dw
import csv, os

INDEXER      = 'jaredfern/vecshare-indexer'
INDEX_FILE   = 'index_file'
EMB_TAG      = 'vecshare'

BASE_URL     = 'https://data.world/'
INDEXER_URL  = BASE_URL + INDEXER  # URL for index & signature file
DATASETS_URL = 'https://data.world/datasets/' + EMB_TAG           # URL for uploaded embeddings
DW_CLASS_TAG = 'dw-dataset-name DatasetCard__name___2U4-H'

def refresh_indexer(force_update = False):
    # Retrieve source for data.world:vecshare search results
    wd = webdriver.Chrome()
    wd.get(DATASETS_URL)
    try:
        WebDriverWait(wd,5).until(EC.visibility_of_element_located((By.CLASS_NAME, DW_CLASS_TAG)))
    except: pass

    # Scrape source for dataset links
    soup = BeautifulSoup(wd.page_source, 'lxml')
    sets    = [s["href"][1:] for s in soup.find_all('a', DW_CLASS_TAG)]
    dw_api = dw.api_client()
    wd.close()

    header = [
        "Embedding Name",
        "Dataset Name",
        "Contributor",
        "Embedding Type",
        "Dimension",
        "Vocab Size",
        "Case Sensitive",
        "File Format",
        "Download URL"
    ]

    header_check = set([lower(field.replace(' ', '_')) for field in header])
    index_fields = set(dw.query(INDEXER, 'SELECT * FROM '+ INDEX_FILE).dataframe)
    # prev_sets    =

    if (force_update or header_check != index_fields):

    with open(INDEX_FILE+".csv", 'w') as ind:
        # CUSTOM INDEXERS: Add additional metadata fields to header file
        csv_writer = csv.DictWriter(ind, fieldnames = header)
        csv_writer.writeheader()
        for s in sets:
            curr_set  = dw.load_dataset(s,force_update = True) # Embedding
            curr_meta = dw_api.get_dataset(s)
            contrib   = curr_meta["owner"]
            meta_dict = dict()
            for field in curr_meta["summary"].split(","):
                meta_field = field.split(":")
                meta_dict[meta_field[0].strip()] = meta_field[1].strip()

            for each in curr_set.tables:
                curr_emb = curr_set.describe(each)
                emb_dim = len(curr_emb['schema']['fields']) - 1
                file_format = curr_emb['format']
                vocab_size = dw.query(s , "SELECT COUNT(*) FROM " + each).dataframe.iloc[0][0]

                print "Indexed embedding: " + s + " from dataset " + each + "."
                meta_dict.update({
                            'Embedding Name': each,
                            "Dataset Name":s,
                            "Contributor":contrib,
                            "Dimension":emb_dim,
                            "Vocab Size":vocab_size,
                            "File Format":file_format})
                csv_writer.writerow(meta_dict)

    print "Updating index file at " + INDEXER_URL
    dw_api.upload_files(INDEXER, os.getcwd() + '/'+ INDEX_FILE +'.csv')
