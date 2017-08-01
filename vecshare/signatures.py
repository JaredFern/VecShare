import pandas as pd
import datadotworld as dw
import os,string,re,codecs,requests,indexer,io
from nltk.tokenize import sent_tokenize,word_tokenize
from collections import Counter
from operator import itemgetter

try:
    from StringIO import StringIO
    import cPickle as pickle
except: import io, pickle
"""
    File containing signature similarity measures and aassociated helpers.
    Implemented similarity measures:
        AvgRank Similarity
"""
# AvgRank Signature Similarity Method
def avgrank(inp_dir):
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
    test_vocab = avgrank_corp(inp_dir,stopwords)
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
def avgrank_corp(inp_dir,hdv_vocab, num = 5000):
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
