from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from bs4 import BeautifulSoup
from dateutil.parser import parse
from copy import deepcopy
from tabulate import tabulate
import datadotworld as dw
import pandas as pd
import csv,os,datetime,requests,string,sys,re,pdb
from pprint import pprint
try:
    from StringIO import StringIO
    import cPickle as pickle
    import sim_benchmark
    import info,vecshare
except:
    import io, pickle
    import vecshare.sim_benchmark as sim_benchmark
    import vecshare.vecshare as vecshare
    import vecshare.info as info

def refresh(force_update=False):
    '''
    Crawls for new embeddings with the tag and update the index file with new
    embedding sets, or changes to existing shared embeddings.

    Args:
        force_update(bool, opt): Hard reset, re-index ALL available embeddings.
            If False, only scrape metadata or new embedding sets.
    Returns:
        None. Uploads new index_file.csv to indexer on data store.
    '''
    # Retrieve source for data.world:vecshare search results
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    wd = webdriver.Chrome()
    wd.get(info.DATASETS_URL)
    try:
        WebDriverWait(wd,5).until(EC.visibility_of_element_located((By.CLASS_NAME, info.DW_CLASS_TAG)))
    except: pass

    soup    = BeautifulSoup(wd.page_source, 'lxml')
    sets    = [s["href"][1:] for s in soup.find_all('a', info.DW_CLASS_TAG)]
    dw_api  = dw.api_client()
    wd.close()
    print ("Found " + str(len(sets)) + " sets with the " + info.EMB_TAG + " tag.")

    embeddings, prev_indexed, updated = [], [], False
    if not force_update:
        prev_query = dw.query(info.INDEXER, 'SELECT dataset_name, embedding_name FROM '+ info.INDEX_FILE).dataframe
        for ind, row in prev_query.iterrows():
            prev_indexed.append("/".join(row.values))
    for set_name in sets:
        curr_set  = dw.load_dataset(set_name,force_update = True) # Embedding
        curr_meta = dw_api.get_dataset(set_name)
        set_updated = parse(curr_meta['updated'])
        meta_dict = dict()
        contrib   = curr_meta["owner"]
        resources = curr_set.describe()['resources']

        summary = StringIO(curr_meta["summary"])
        for line in summary:
            for field in line.split(","):
                for sent in field.split("."):
                    try:
                        meta_field = field.split(":")
                        if len(meta_field) == 2:
                            meta_dict[meta_field[0].strip().lower().replace(" ", "_")] = meta_field[1].strip()
                    except: pass

        for each in curr_meta['files']:
            emb_name = each['name'][:-4]
            emb_updated = parse(each['updated'])
            try:
                ind_query = 'SELECT last_updated FROM '+ info.INDEX_FILE + \
                ' WHERE dataset_name = "'+ set_name +'" and embedding_name = "'+emb_name+'"'
                query_results = dw.query(info.INDEXER, ind_query).dataframe.iloc[0].values[0]
                last_indexed = parse(query_results)
                last_updated = emb_updated if emb_updated > set_updated else set_updated
            except:
                last_updated = datetime.datetime.now()
                pass

            # Index if new embedding or if metadata/embedding updated since last Index
            if (force_update) or (set_name + '/' + emb_name not in prev_indexed) or (last_indexed < last_updated):
                try: curr_emb = curr_set.describe(emb_name.lower())
                except: continue
                updated = True
                emb_dim = len(curr_emb['schema']['fields']) - 1
                file_format = curr_emb['format']
                vocab_size = dw.query(set_name , "SELECT COUNT(text) FROM " + emb_name).dataframe.iloc[0][0]
                emb_simset = vecshare.extract(emb_name,'sim_vocab', set_name=set_name, case_sensitive=True,progress=False)
                score_dict  = sim_benchmark._eval_all(emb_simset)

                temp_0  ='original/'+emb_name.lower()+'.csv'
                temp_1  =emb_name.lower()

                for d in resources:
                    if d['name'] == temp_0:
                        try:
                            description = StringIO(d['description'])
                            for line in description:
                                for sent in line.split("."):
                                    for field in sent.split(","):
                                        meta_field = field.split(":")
                                        if len(meta_field) == 2:
                                            meta_dict[meta_field[0].strip().lower().replace(" ", "_")] = meta_field[1].strip()
                        except: pass
                    if d['name'] == temp_1:
                        try:
                            description = StringIO(d['description'])
                            for line in description:
                                for sent in line.split('.'):
                                    for field in sent.split(","):
                                        meta_field = field.split(":")
                                        if len(meta_field) == 2:
                                            meta_dict[meta_field[0].strip().lower().replace(" ", "_")] = meta_field[1].strip()
                        except: pass
                print ("Newly Indexed embedding: " + emb_name+ " from dataset " + set_name + ".")
                meta_dict.update(score_dict)
                meta_dict.update({
                            u'embedding_name': emb_name,
                            u"dataset_name": set_name,
                            u"contributor":contrib,
                            u"dimension":emb_dim,
                            u"vocab_size":vocab_size,
                            u"file_format":file_format,
                            u"last_updated": last_updated})
                embeddings.append(deepcopy(meta_dict))
                pprint (meta_dict)
            else:
                print ("Re-indexed embedding: " + emb_name+ " from dataset " + set_name + ".")
                query = 'SELECT * FROM '+ info.INDEX_FILE + ' WHERE dataset_name = "'+ \
                set_name +'" and embedding_name = "'+ emb_name +'"'
                prev_row = dw.query(info.INDEXER, query).dataframe
                embeddings.extend(prev_row.to_dict(orient='records'))

    with open(info.INDEX_FILE+".csv", 'w') as ind:
        meta_header = set().union(*embeddings)
        csv_writer = csv.DictWriter(ind, fieldnames = meta_header)
        csv_writer.writeheader()
        for emb in embeddings:
            csv_writer.writerow(emb)

    print ("Updating index file at " + info.INDEXER_URL)
    dw_api.upload_files(info.INDEXER, os.getcwd() + '/'+ info.INDEX_FILE +'.csv')
    if updated:
        _emb_rank()
        print ("Updating avg_rank signatures")
        avgrank_refresh()

def _emb_rank():
    query = 'SELECT embedding_name, dataset_name, contributor, embedding_type, dimension, score \
        FROM ' + info.INDEX_FILE
    results = dw.query(info.INDEXER,query).dataframe
    results = results.nlargest(10, 'score')
    for ind,row in results.iterrows():
        results.loc[ind, 'embedding_name'] = \
        "[`"+row['embedding_name']+"`]("+ info.BASE_URL + row['dataset_name'] + "/"+ row["embedding_name"] +")"

    results.drop('dataset_name', axis=1)
    md_table = tabulate(results, headers=list(results), tablefmt="pipe",showindex=False)
    with open('../README.md', 'r') as readme:
        pre, post = True,False
        pre_table,post_table = '',''
        for line in readme:
            if pre:
                pre_table += line
                if line == '[comment]: <> (Leaderboard Start)\n':
                    pre = False
            if post: post_table+= line
            if line == '[comment]: <> (Leaderboard End)\n':
                post_table = line
                post = True
    with open('../README.md', 'w') as readme:
        readme.write(pre_table+'\n')
        readme.write(md_table+'\n\n')
        readme.write(post_table)

# Determines the n most frequent stop words, found in at least 'tolerance' fraction of the embeddings.
def avgrank_refresh(tolerance = 0.60,sig_cnt = 5000,stopword_cnt = 100):
    '''
    If there are changes to the set of shared embeddings, refresh the AvgRank signature.

    Generate a set of at most `stopword_cnt` stopwords that occur in at least
    `tolerance` * emb_cnt embeddings. Generate signatures for the embeddings
    using the `sig_cnt` most common remaining words.

    Args:
        tolerance (float): Frequency at which a stopword must occur
        sig_cnt (int): Size of AvgRank signature vocab_size
        stopword_cnt (int): Max number of stopwords

    Returns:
        None. Uploads new ar_sig.txt (serialized signatures) to data store.
    '''
    stopwords, emb_vocab, signatures = [],{}, {}
    DW_API_TOKEN = os.environ['DW_AUTH_TOKEN']

    emb_list = dw.query(info.INDEXER, 'SELECT embedding_name, dataset_name FROM ' + info.INDEX_FILE).dataframe
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
    dw_api.upload_files(info.SIGNATURES, os.getcwd() + '/ar_sig.txt')
