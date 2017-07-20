import pandas as pd
import numpy as np
import datadotworld as dw
import urllib,urllib2,requests,sys,os,zipfile,string, indexer
from scipy import spatial
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize,word_tokenize
from signatures import *
from embedding import embedding

def check():
	embedding_list = dw.query(indexer.INDEXER_URL, 'SELECT * FROM ' + indexer.INDEX_FILE)
	df = embedding_list.dataframe
	cols  = df.columns.tolist()
	title = ["embedding_name" , "dataset_name", "contributor"]
	return df [title + [field for field in cols if field not in title]]

# Report of the process of downloading large word embeddings.
def report(count, blockSize, totalSize):
	percent = float(count*blockSize*100/totalSize)
	sys.stdout.write("\r%d%%" % percent + ' complete')
	sys.stdout.flush()

# SQL query for specific word embedding in given table
def query(words, emb_name, set_name = None):
	emb_list = dw.query(indexer.INDEXER_URL, 'SELECT * FROM ' + indexer.INDEX_FILE).dataframe
	emb_names = emb_list.embedding_name

	if len(emb_list.loc[emb_list['embedding_name'] == emb_name]) > 1:
		raise ValueError("More than one embedding exists with name: " + emb_name + "Try again specifying set_name.")
	if emb_name not in emb_names.values:
		raise ValueError("No embedding exists with name: " + emb_name)
	if emb_list.file_format[emb_names == emb_name].iloc[0] not in ['csv', 'tsv', 'xls', 'rdf', 'json']:
		raise TypeError(emb_name + " was not uploaded in a queryable format. Try reuploading as: csv, tsv, xls, rdf, or json.")

	if not set_name:
		set_name = emb_list.loc[emb_list['embedding_name'] == emb_name].dataset_name.iloc[0]

	first_col = ['column_a', 'the']
	for title in first_col:
		try:
			if len(words)>1:
				query = 'SELECT * FROM ' + emb_name + ' where ' + title + ' in' + str(tuple(words))
			else:
				query = 'SELECT * FROM ' + emb_name + ' where ' + title + 'column_a =' + words[0]
			results = dw.query(set_name, query).dataframe
			results = results[results.columns[::-1]]
			if len(results) < len(words): continue
			return results.values.tolist()
		except: pass

	words = [word.lower() for word in words]
	first_col = ['lower(column_a)', 'lower(the)']
	print ("No case sensitive match found. Searching for non-case sensitive match.")
	for title in first_col:
		try:
			if len(words)>1:
				query = 'SELECT * FROM ' + emb_name + ' where ' + title + ' in' + str(tuple(words))
			else:
				query = 'SELECT * FROM ' + emb_name + ' where ' + title + 'column_a =' + words[0]
			results = dw.query(set_name, query).dataframe
			results = results[results.columns[::-1]]
			return results.values.tolist()
		except: pass
	raise RuntimeError("No matching word vector found.")

# Get the subset of the pretrained embeddings according to the raw text input.
def EmbedExtract(file_dir,table,batch = 200,pad = False,check = False,download = False):

	#Get the indexer of all available embeddings.
	embedding_list = pd.read_csv(indexer.broker_url)
	table_list = embedding_list['table']
	format_list = embedding_list['file_format']
	name_list = embedding_list['embedding_name']

	#Check whether the embedding user asked for exist or not.
	if not name_list[table_list == table].values:
		print "The embedding you asked to extract from doesn't exist."
		return
	if format_list[table_list == table].values[0] != 'csv':
		print 'Sorry for the inconvenience but we are not able to query Non-csv file.'
		return

	texts = ''
	for name in sorted(os.listdir(file_dir)):
		path = os.path.join(file_dir, name)
		if os.path.isdir(path):
			for fname in sorted(os.listdir(path)):
				fpath = os.path.join(path, fname)
				if sys.version_indexer < (3,):
					f = open(fpath)
				else:
					f = open(fpath, encoding='latin-1')
				texts = texts + f.read()
	input_txt = []
	sentences = sent_tokenize(texts)
	for s in sentences:
		tokens = word_tokenize(s)
		input_txt = input_txt + tokens
	inp_vocab = set(input_txt)
	inp_vsize = (len(inp_vocab))

	dataset_ = indexer.table_parser
	query_ = ''
	final_result = []
	back_query = ''
	print 'Embedding extraction begins.'
	words = list(inp_vocab)
	i = 0
	back_up_i = 0

	#Extraction is able to recover from Runtime error by adding restore mechanism.
	while i < len(words):
		if i == 0:
			query_ = 'SELECT * FROM ' + table + " where `Column A` = '" + words[i] + "'"
		elif i % batch == 0:
			process = str((i)*100/len(words))
			try:
				results = dw.query(dataset_, query_)
				print process + "%" + ' has been completed.'
				back_query = query_
				back_up_i = i
			except RuntimeError,e:
				i = back_up_i
				print 'Batch size ' + str(batch)
				batch = int(batch * 0.9)
				if batch == 0:
					print "404 Error"
					break
				print 'Batch size too large. Reducing to ' + str(batch)
				continue
			vector = results.dataframe.values
			final_result.extend(vector)
			query_ = 'SELECT * FROM ' + table + " where `Column A` = '" + words[i] + "'"
		else:
			query_ = query_ + " OR `Column A` = '" + words[i] + "'"
		i = i + 1

	if batch == 0:
		print "Embedding extraction failed."
		return

	print 'Embedding successfully extracted.'
	word_vector = {}
	ans = ''

	for vector in final_result:
		word = str(vector[0])
		embed = np.array2string(vector[1:])[1:-1]
		embed = embed.replace('\n','')
		word_vector[word] = embed
		ans = ans + word + ' ' + embed + '\n'
	overlap_words = []

	if pad == True:
		for word in inp_vocab:
			if not word_vector.has_key(word):
				subset = []
				subset.append(word.encode('utf-8'))
				zero = np.zeros(final_result[0].shape[0]-1)
				subset.extend(zero)
				subset = np.asarray(subset,dtype=object)
				final_result.append(subset)
				embed = np.array2string(zero)[1:-1]
				ans = ans + word + ' ' + embed + '\n'

	if download == True:
		f = open('Extracted_' + table + '.txt','w')
		pool = ans.split('\n')
		for line in pool:
			f.write(line)
			f.write('\n')
		f.close()

	for word in word_vector:
		overlap_words.append(word)
	int_count = int(len(set.intersection(set(overlap_words),set(words))))
	missing_words = str(len(words) - int_count)
	percent = str(int_count * 100 / len(words))

	print 'There are ' + missing_words + " tokens that can't be found in this pretrained word embedding."
	print percent + "%" + " words can be found in this pretrained word embedding."

	if check:
		print ans
	return final_result
