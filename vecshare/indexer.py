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
import io,os,datetime,requests,string,sys,re,pytz
import unicodecsv as csv

from pyvirtualdisplay import Display
try:
    from StringIO import StringIO
    import cPickle as pickle
    import sim_benchmark
    import info,vecshare
except:
    import pickle
    from io import StringIO
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
    display = Display(visible=0, size=(800, 600))
    display.start()
    wd= webdriver.Firefox(executable_path="/usr/bin/firefox", capabilities= {"marionette": False })

    sets, upload_types = {}, [info.SMALL_EMB_TAG, info.LARGE_EMB_TAG]
    for emb_size in upload_types:
        page_num, set_count, sets[emb_size] = 1, sys.maxsize,[]
        while set_count > len(sets[emb_size]):
            wd.get(info.DATASETS_URL + emb_size + "?page="+str(page_num))
            try:
                WebDriverWait(wd,5).until(EC.visibility_of_element_located((By.CLASS_NAME, info.DW_CLASS_TAG)))
            except: pass
            soup    = BeautifulSoup(wd.page_source, 'lxml')
            set_txt = soup.find('h1','TopicView__headline___2_0-1').text
            set_count = [int(s) for s in set_txt.split() if s.isdigit()][0]
            sets[emb_size].extend([s["href"][1:] for s in soup.find_all('a', info.DW_CLASS_TAG)])
            page_num += 1

    dw_api  = dw.api_client()
    wd.close()
    print ("Found " + str(len(sets[info.SMALL_EMB_TAG]) + len(sets[info.LARGE_EMB_TAG])) + " sets with the vecshare tags.")

    embeddings, prev_indexed, updated = [], [], False
    if not force_update:
        prev_query = dw.query(info.INDEXER, 'SELECT dataset_name, embedding_name FROM '+ info.INDEX_FILE).dataframe
        for ind, row in prev_query.iterrows():
            prev_indexed.append("/".join(row.values))

    for set_size, set_names in sets.items():
        for set_name in set_names:
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
                                meta_dict[meta_field[0]\
                                    .strip()\
                                    .lower().replace(" ", "_")\
                                    .replace("-", "_")]\
                                    = meta_field[1].strip()
                        except: pass
            for each in curr_meta['files']:
                emb_name = each['name'][:-4]
                emb_updated = parse(each['updated'])
                try:
                    ind_query = 'SELECT last_updated FROM '+ info.INDEX_FILE + \
                    ' WHERE dataset_name = "'+ set_name +'" and embedding_name = "'+emb_name+'"'
                    query_results = dw.query(info.INDEXER, ind_query).dataframe.iloc[0].values[0]
                    last_indexed = parse(query_results)
                    if emb_updated > set_updated: last_updated = emb_updated
                    else: last_updated =  set_updated
                except:
                    last_updated = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)
                    last_indexed = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)
                    pass
                # Index if new embedding or if metadata/embedding updated since last Index
                if (force_update) or (set_name + '/' + emb_name not in prev_indexed) or  (last_indexed < last_updated) :
                    try:
                        curr_emb = curr_set.describe(emb_name.lower())
                        file_format = curr_emb['format']
                    except:
                        file_format ='.csv'
                        continue
                    updated = True

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
                    if set_size == info.LARGE_EMB_TAG:
                        _appx_count = int(meta_dict['app_num'])
                        emb_simset = vecshare.extract(emb_name+"_appx0", os.path.join('vecshare','sim_vocab'), \
                            set_name=(set_name.split("/")[0]+'/'+emb_name+"-appx0").replace("_","-"), case_sensitive=True,vs_format='small')
                    if set_size == info.SMALL_EMB_TAG:
                        emb_simset = vecshare.extract(emb_name, '/home/jared/VecShare/vecshare/sim_vocab', \
                            set_name=set_name, case_sensitive=True,vs_format='small')
                    try:
                        score_dict  = sim_benchmark._eval_all(emb_simset)
                        meta_dict.update(score_dict)
                    except: pass

                    print ("Newly Indexed embedding: " + emb_name+ " from dataset " + set_name + ".")
                    meta_dict.update({
                                u'embedding_name': emb_name,
                                u"dataset_name": set_name,
                                u"contributor":contrib,
                                u"file_format":file_format,
                                u"last_updated": last_updated,
                                u"vs_format": 'large' if set_size == info.LARGE_EMB_TAG else 'small'})
                    if set_size == info.SMALL_EMB_TAG:
                        meta_dict.update({"app_num": "None"})
                    embeddings.append(deepcopy(meta_dict))
                else:
                    print ("Re-indexed embedding: " + emb_name+ " from dataset " + set_name + ".")
                    query = 'SELECT * FROM '+ info.INDEX_FILE + ' WHERE dataset_name = "'+ \
                    set_name +'" and embedding_name = "'+ emb_name +'"'
                    prev_row = dw.query(info.INDEXER, query).dataframe
                    embeddings.extend(prev_row.to_dict(orient='records'))

    with io.open(info.INDEX_FILE_PATH, 'wb') as ind:
        meta_header = set().union(*embeddings)
        csv_writer = csv.DictWriter(ind, fieldnames = meta_header)
        csv_writer.writeheader()
        for emb in embeddings: csv_writer.writerow(emb)

    print ("Updating index file at " + info.INDEXER_URL)
    dw_api.upload_files(info.INDEXER, info.INDEX_FILE_PATH)
    if updated:
        print ("Updating avg_rank signatures")
        avgrank_refresh()
        _emb_rank()
        return updated
    else: return False

def _emb_rank():
    query = 'SELECT embedding_name, dataset_name, contributor, embedding_type, score \
        FROM ' + info.INDEX_FILE
    results = dw.query(info.INDEXER,query).dataframe
    results = results.nlargest(10, 'score')
    print results
    import pdb; pdb.set_trace()
    for ind,row in results.iterrows():
        results.loc[ind, 'embedding_name'] = \
        "[`"+row['embedding_name']+"`]("+ info.BASE_URL + row['dataset_name']+")"

    results = results.drop('dataset_name', axis=1)
    md_table = tabulate(results, headers=list(results), tablefmt="pipe",showindex=False)
    with io.open('/home/jared/VecShare/README.md', 'r', encoding='utf-8') as readme:
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
    with io.open('/home/jared/VecShare/README.md', 'wb') as readme:
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

    #emb_list = dw.query(info.INDEXER, 'SELECT embedding_name, dataset_name FROM ' + info.INDEX_FILE).dataframe
    emb_list = pd.read_csv(info.INDEX_FILE_PATH)
    threshold = int(0.5 + tolerance * emb_list.shape[0])
    for ind, row in emb_list.iterrows():
        if row['vs_format'] == 'large':
            emb_name, set_name = row['embedding_name']+"-appx0", row['dataset_name']+"-appx0"
        else:
            emb_name, set_name = row['embedding_name'], row['dataset_name']
        query_url = "https://query.data.world/file_download/"+set_name+"/"+ emb_name + '.csv'
        payload, headers = "{}", {'authorization': 'Bearer '+ DW_API_TOKEN}
        emb_text = StringIO(requests.request("GET", query_url, data=payload, headers=headers).text)
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

    pickle.dump(signatures, io.open(info.AR_SIG_PATH, "wb"))
    dw_api  = dw.api_client()
    print ("Uploading AvgRank signatures")
    dw_api.upload_files(info.SIGNATURES, info.AR_SIG_PATH)
