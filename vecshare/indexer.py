from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium import webdriver
from bs4 import BeautifulSoup
from dateutil.parser import parse
from copy import deepcopy
import datadotworld as dw
import pandas as pd
import csv,os,datetime,requests,string

try: 
    from StringIO import StringIO
    import cPickle as pickle
except: import io, pickle
INDEXER      = 'jaredfern/vecshare-indexer'
INDEX_FILE   = 'index_file'
EMB_TAG      = 'vecshare'
SIGNATURES   = 'jaredfern/vecshare-signatures'

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
        u"embedding_name",
        u"dataset_name",
        u"contributor",
        u"embedding_type",
        u"dimension",
        u"vocab_size",
        u"case_sensitive",
        u"file_format",
        u"last_updated"
    ]
    embeddings, prev_indexed, updated = [], [], False
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
            meta_dict[meta_field[0].strip().lower().replace(" ", "_")] = meta_field[1].strip()

        for each in curr_meta['files']:
            emb_name = each['name'][:-4]
            emb_updated = parse(each['updated'])
            try:
                ind_query = 'SELECT last_updated FROM '+ INDEX_FILE + \
                ' WHERE dataset_name = "'+ set_name +'" and embedding_name = "'+emb_name+'"'
                query_results = dw.query(INDEXER, ind_query).dataframe.iloc[0].values[0]
                last_indexed = parse(query_results)
                last_updated = emb_updated if emb_updated > set_updated else set_updated
            except: pass

            # Index if new embedding or if metadata/embedding updated since last Index
            if (force_update) or (set_name + '/' + emb_name not in prev_indexed) or (last_indexed < last_updated):
                try: curr_emb = curr_set.describe(emb_name.lower())
                except: continue
                updated = True
                emb_dim = len(curr_emb['schema']['fields']) - 1
                file_format = curr_emb['format']
                vocab_size = dw.query(set_name , "SELECT COUNT(text) FROM " + emb_name).dataframe.iloc[0][0]

                print ("Newly Indexed embedding: " + emb_name+ " from dataset " + set_name + ".")
                meta_dict.update({
                            u'embedding_name': emb_name,
                            u"dataset_name": set_name,
                            u"contributor":contrib,
                            u"dimension":emb_dim,
                            u"vocab_size":vocab_size,
                            u"file_format":file_format,
                            u"last_updated": last_updated})
                embeddings.append(deepcopy(meta_dict))
            else:
                print ("Re-indexed embedding: " + emb_name+ " from dataset " + set_name + ".")
                query = 'SELECT * FROM '+ INDEX_FILE + ' WHERE dataset_name = "'+ \
                set_name +'" and embedding_name = "'+ emb_name +'"'
                prev_row = dw.query(INDEXER, query).dataframe
                embeddings.extend(prev_row.to_dict(orient='records'))

    with open(INDEX_FILE+".csv", 'w') as ind:
        meta_header = set().union(*embeddings)
        csv_writer = csv.DictWriter(ind, fieldnames = meta_header)
        csv_writer.writeheader()
        for emb in embeddings:
            csv_writer.writerow(emb)

    print ("Updating index file at " + INDEXER_URL)
    dw_api.upload_files(INDEXER, os.getcwd() + '/'+ INDEX_FILE +'.csv')
    if updated:
        print ("Updating avg_rank signatures")
        avgrank_refresh()


# Determines the n most frequent stop words, found in at least 'tolerance' fraction of the embeddings.
def avgrank_refresh(tolerance = 0.60,sig_cnt = 5000,stopword_cnt = 100):
    stopwords, emb_vocab, signatures = [],{}, {}
    DW_API_TOKEN = os.environ['DW_AUTH_TOKEN']

    emb_list = dw.query(INDEXER, 'SELECT embedding_name, dataset_name FROM ' + INDEX_FILE).dataframe
    threshold = int(0.5 + tolerance * emb_list.shape[0])
    for ind, row in emb_list.iterrows():
        emb_name, set_name = row['embedding_name'], row['dataset_name']
        query_url = "https://query.data.world/file_download/"+set_name+"/"+ emb_name + '.csv'
        payload, headers = "{}", {'authorization': 'Bearer '+ DW_API_TOKEN}
        if sys.version_info < (3,):
            emb_text = StringIO(requests.request("GET", query_url, data=payload, headers=headers).text)
        else:
            emb_text = io.StringIO(requests.request("GET", query_url, data=payload, headers=headers).text)

        emb_df = pd.read_csv(emb_text, nrows = 1.5 *sig_cnt)

        wordlist = emb_df.iloc[0:2*stopword_cnt,0].values
        signatures.update({emb_name: emb_df.iloc[:,0].values})
        for word in wordlist:
            word = str(word).lower()
            if (word not in emb_vocab): emb_vocab.update({word: 1})
            else:emb_vocab[word] += 1

    stopwords.extend(list(string.digits))
    stopwords.extend(list(string.punctuation))
    for key in emb_vocab:
        if (emb_vocab[key] >= threshold): stopwords.append(key)

    for emb_name, emb_sig  in signatures.items():
        emb_sig = emb_sig.tolist()
        for word in stopwords:
            if word in emb_sig: emb_sig.remove(word)
        emb_sig = emb_sig[:sig_cnt]
        print ("Generated AvgRank signature for: " + emb_name)
        signatures.update({emb_name:emb_sig})
    signatures.update({'stopwords':stopwords})

    pickle.dump(signatures, open( "ar_sig.txt", "w" ))
    dw_api  = dw.api_client()
    print ("Uploading AvgRank signatures")
    dw_api.upload_files(SIGNATURES, os.getcwd() + '/ar_sig.txt')
