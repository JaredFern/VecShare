import pandas as pd
import numpy as np
import datadotworld as dw
import os,string,re,codecs,requests,io,sys
from nltk.tokenize import sent_tokenize,word_tokenize
from collections import Counter
from operator import itemgetter

try:
    from StringIO import StringIO
    import cPickle as pickle
    import info
except ImportError:
    import io, pickle
    import vecshare.info as info

"""
    File containing signature similarity measures and associated helpers.
    Implemented similarity measures:
        AvgRank Similarity
        Analogy Score
"""
def simscore(test_set="score"):
    '''
    Select the embedding scoring highest on the `test_set` word comparison task.
    Score is calculated by measuring the average Spearman correlation of the word
    vector cosine similarities and human-rated similarity for each word pair.
    Missing words are substituted with random words

    Args:
        test_set (str,opt): Embedding will be returned with highest score on `test_set`
            If test_set is not specified, embedding will be returned with highest
            average score over all similarity tasks (excluding MC-30)

    Returns:
        emb_name: Name of Embedding with highest simscore

    Word Pair Similarity Tasks:
    * WS-353: Finkelstein et. al, 2002
    * MC-30: Miller and Charles, 1991
        (Excluded from composite score, already included in SimLex-999 Test Set)
    * MEN: Bruni et. al, 2012
    * MTurk-287: Radinsky et. al, 2011
    * MTurk-771: Halawi and Dror, 2012
    * Rare-Word: Luong et. al, 2013
    * SimLex-999: Hill et. al, 2014
    * SimVerb-3500: Gerz et. al, 2016
    * Verb-144: Baker et. al, 2014
    '''
    max_query = "SELECT MAX(" + test_set +") FROM " +info.INDEX_FILE
    max_simscore = np.floor(100*dw.query(info.INDEXER, max_query).dataframe.iloc[0][0])/100
    emb_query = "SELECT embedding_name FROM " + info.INDEX_FILE + " WHERE " +test_set +" >= " + str(max_simscore)
    top_emb = dw.query(info.INDEXER, emb_query).dataframe
    return top_emb.iloc[0][0]

# AvgRank Signature Similarity Method
def avgrank(inp_dir):
    '''
    Returns the most similar embedding in terms of vocab and frequency overlap with the user corpus.

    Args:
        inp_dir(str): Path to user's corpus

    Returns:
        emb_name(str): Name of most similar embedding
    '''
    rank_dict = dict()
    DW_API_TOKEN = os.environ['DW_AUTH_TOKEN']
    query_url = "https://query.data.world/file_download/jaredfern/vecshare-signatures/ar_sig.txt"
    payload, headers = "{}", {'authorization': 'Bearer '+ DW_API_TOKEN}
    if sys.version_info < (3,):
        emb_text = StringIO(requests.request("GET", query_url, data=payload, headers=headers).text)
    else:
        emb_text = io.StringIO(requests.request("GET", query_url, data=payload, headers=headers).text)

    signatures = pickle.load(emb_text)

    stopwords  = signatures.pop('stopwords', None)
    test_vocab = _avgrank_corp(inp_dir,stopwords)
    for emb_name,emb_sig in signatures.items():
        rank_dict.update({emb_name: 0})
        for ind in range(0,len(signatures[emb_name])):
            curr_inpword = signatures[emb_name][ind]
            if curr_inpword in test_vocab:
                rank_dict[emb_name] += test_vocab.index(curr_inpword)/float(len(test_vocab))
            else:
                rank_dict[emb_name] += len(signatures[emb_name])/float(len(test_vocab))

    ranked_embs = sorted(rank_dict.items(),key=itemgetter(1))
    return ranked_embs[0][0]

# Generates the 5000 most frequent words in the test corpus from the txt file
def _avgrank_corp(inp_dir,hdv_vocab, num = 5000):
    cnt, vocab = Counter(), []
	# Counter for all words in the corpus
    for (root, dirs, files) in os.walk(inp_dir):
        files = [f for f in files if not f[0] == '.']
        for f in files:
            filepath = os.path.join(root,f)
            with codecs.open(filepath,'r', encoding="utf-8") as f:
                tok_txt = word_tokenize(f.read())
                for word in tok_txt: cnt[word] += 1
    for word in hdv_vocab:
        if word in cnt.keys(): del cnt[word]
    for word in cnt.most_common(num):
        try:    vocab.append(str(word[0]))
        except:	continue
    return vocab
