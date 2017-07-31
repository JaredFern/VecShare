from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium import webdriver
from bs4 import BeautifulSoup
from dateutil.parser import parse
from copy import deepcopy
import datadotworld as dw
import csv,os,datetime

INDEXER      = 'jaredfern/vecshare-indexer'
INDEX_FILE   = 'index_file'
EMB_TAG      = 'vecshare'

BASE_URL     = 'https://data.world/'
INDEXER_URL  = BASE_URL + INDEXER  # URL for index & signature file
DATASETS_URL = 'https://data.world/datasets/' + EMB_TAG           # URL for uploaded embeddings
DW_CLASS_TAG = 'dw-dataset-name DatasetCard__name___2U4-H'

def refresh(force_update=False):
    # Retrieve source for data.world:vecshare search results
    wd = webdriver.Chrome()
    wd.get(DATASETS_URL)
    try:
        WebDriverWait(wd,5).until(EC.visibility_of_element_located((By.CLASS_NAME, DW_CLASS_TAG)))
    except: pass

    soup    = BeautifulSoup(wd.page_source, 'lxml')
    sets    = [s["href"][1:] for s in soup.find_all('a', DW_CLASS_TAG)]
    dw_api  = dw.api_client()
    wd.close()
    print ("Found " + str(len(sets)) + " sets with the " + EMB_TAG + " tag.")

    base_header = [
        "Embedding Name",
        "Dataset Name",
        "Contributor",
        "Embedding Type",
        "Dimension",
        "Vocab Size",
        "Case Sensitive",
        "File Format",
        "Download URL",
        "Last Updated"
    ]
    embeddings, prev_indexed = [], []
    prev_query = dw.query(INDEXER, 'SELECT dataset_name, embedding_name FROM '+ INDEX_FILE).dataframe
    for ind, row in prev_query.iterrows():
        prev_indexed.append("/".join(row.values))

    for set_name in sets:
        curr_set  = dw.load_dataset(set_name,force_update = True) # Embedding
        curr_meta = dw_api.get_dataset(set_name)
        set_updated = parse(curr_meta['updated'])
        meta_dict = dict()
        contrib   = curr_meta["owner"]

        for field in curr_meta["summary"].split(","):
            meta_field = field.split(":")
            meta_dict[meta_field[0].strip()] = meta_field[1].strip()

        for each in curr_meta['files']:
            emb_name = each['name'][:-4].lower()
            emb_updated = parse(each['updated'])

            ind_query = 'SELECT last_updated FROM '+ INDEX_FILE + \
            ' WHERE dataset_name = "'+ set_name +' and embedding_name = "'+emb_name+'"'
            last_indexed = parse(dw.query(INDEXER, ind_query).dataframe.iloc[0])
            last_updated = each['updated'] if emb_updated > set_updated else curr_meta['updated']

            # Index if new embedding or if metadata/embedding updated since last Index
            if (force_update) or (set_name + '/' + emb_name not in prev_indexed) or (last_indexed < last_updated):
                try: curr_emb = curr_set.describe(emb_name)
                except: continue

                emb_dim = len(curr_emb['schema']['fields']) - 1
                file_format = curr_emb['format']
                vocab_size = dw.query(set_name , "SELECT COUNT(*) FROM " + emb_name).dataframe.iloc[0][0]

                print "Indexed embedding: " + emb_name+ " from dataset " + set_name + "."
                meta_dict.update({
                            'Embedding Name': emb_name,
                            "Dataset Name": set_name,
                            "Contributor":contrib,
                            "Dimension":emb_dim,
                            "Vocab Size":vocab_size,
                            "File Format":file_format,
                            "Last Updated": last_updated})
                embeddings.append(deepcopy(meta_dict))

            # If no changes, copy previously indexed metadata
            else:
                query = 'SELECT * FROM '+ INDEX_FILE + "WHERE dataset_name = "+ \
                set_name +' and embedding_name = '+ emb_name
                prev_row = dw.query(INDEXER, query).dataframe.to_dict()
                embeddings.append(prev_row)

    with open(INDEX_FILE+".csv", 'w') as ind:
        # CUSTOM INDEXERS: Add additional metadata fields to header file
        csv_writer = csv.DictWriter(ind, fieldnames = meta_header)
        csv_writer.writeheader()
        for emb in embeddings:
            csv_writer.writerow(emb)

    print "Updating index file at " + INDEXER_URL
    dw_api.upload_files(INDEXER, os.getcwd() + '/'+ INDEX_FILE +'.csv')
