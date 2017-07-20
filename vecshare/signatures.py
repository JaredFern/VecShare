"""
    File containing signature similarity measures and aassociated helpers.
    Implemented similarity measures:
        AvgRank Similarity
"""
import pandas as pd
import os,string,re,codecs
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize,word_tokenize
from collections import Counter
from operator import itemgetter
import indexer

hdv_vocab = []
# AvgRank Signature Similarity Method
def method_a(inp_dir,num_sig,num_sig_embedding,num_stopwords):
	rank_dict = dict()
	signature,hf_vocab = HighDensityVocab(num = num_sig_embedding,num_stopwords = num_stopwords)
	test_vocab = RankVocabGenerator(inp_dir,num_sig)

	for embed in signature:
		rank_dict.update({embed: 0})
		for ind in range(0,len(signature[embed])):
			curr_inpword = signature[embed][ind]
			if curr_inpword in test_vocab:
				rank_dict[embed] += test_vocab.index(curr_inpword)/float(len(test_vocab))
			else:
				rank_dict[embed] += len(signature[embed])/float(len(test_vocab))
	# pprint (sorted(rank_dict.items(),key=itemgetter(1)))
	best_embedding = sorted(rank_dict.items(),key=itemgetter(1))[0]
	print 'The best pretrained embedding is ' + best_embedding[0] + '.'
    # return key in rank_dict such that value is lowest

# Determines the n most frequent stop words, found in at least 'tolerance' fraction of the embeddings.
def HighDensityVocab(tolerance = 0.85,num = 5000,num_stopwords = 200):
    	embedding_list = pd.read_csv(info.broker_url)
	print 'Infomation of the current avaliable embeddings.'
	print embedding_list
	print str(num_stopwords) + ' most frequent words will be removed as stop words.'
	print "Pick up " + str(num) + ' top frequent words as signature of all avaliable embeddings.'
	url_list = embedding_list['url']
	name_list = embedding_list['embedding_name']
	format_list = embedding_list['file_format']
	files = []
	threshold = int(0.5 + tolerance * len(url_list))
	emb_vocab = {}

	#fetch the embedding from data.world and output the high density vocab for each existing embedding
	for idx,url in enumerate(url_list):
		print 'Fetching '+ name_list.iloc[idx] +' embeddings and creating its signature.'
		if format_list[idx] != 'csv':
			print "Can't fetch embedding because incorrect file format."
			continue
		file = pd.read_csv(url,header = None,error_bad_lines=False)
		wordlist = file.iloc[0:num_stopwords][0].values
		files.append(file)
		for word in wordlist:
			word = word.lower()
			if ( word not in emb_vocab):
				emb_vocab.update({word: 1})
			else:
				emb_vocab[word] += 1
	print 'Processing High Density Vocabulary list.'
	for key in emb_vocab:
		if (emb_vocab[key] >= threshold):
			hdv_vocab.append(key)
	digits = [i for i in string.punctuation]
	punctuation = [i for i in string.punctuation]
	hdv_vocab.extend(digits)
	hdv_vocab.extend(punctuation)

	print "Assembled High Density Vocabulary found in at least ",tolerance *100, " percent of embeddings."

	#Create a dictionary of which keys are embedding names and values are lists of high freq words(signature)
	signature = {}
	for index,file in enumerate(files):
		hf_words = file.iloc[0:num][0].values
		sig_vocab = []
		for word in hf_words:
			word = word.lower()
			if word not in hdv_vocab:
				sig_vocab.append(word)
		signature[name_list.iloc[index]] = sig_vocab
	return signature,hdv_vocab

# Generates the 5000 most frequent words in the test corpus from the txt file
def RankVocabGenerator(inp_dir,num = 5000):
	cnt = Counter()
	# Counter for all words in the corpus
	if os.path.isdir(inp_dir):
		for (root, dirs, files) in os.walk(inp_dir):

			files = [f for f in files if not f[0] == '.']
			for f in files:
				filepath = os.path.join(root,f)
				# CHANGE: Update Codec as needed for test corpus
				try:
					with codecs.open(filepath,'rb', encoding="cp1252") as f:
						decoded_txt = f.read()
						tok_txt = word_tokenize(decoded_txt.lower())
						for word in tok_txt:
							cnt[word] += 1
				except Exception,e:
					continue
		print "Pick top " + str(num) + " most frequently occurring words in the corpus as signature. "
	else:
		print "No Such file or your path is incorrect."
		return
    # Delete HDF Words
    # Update with correct high density vocab filename
	for word in hdv_vocab:
		if word in cnt.keys(): del cnt[word]

	corpus_name = inp_dir.split('/')[-1]
	vocab = []
	for word in cnt.most_common(num):
		try:
			vocab.append(str(word[0]))
		except:
			continue
	return vocab
