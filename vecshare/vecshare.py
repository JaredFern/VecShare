import pandas as pd
import numpy as np
import datadotworld as dw
import urllib,urllib2,requests,sys,os,zipfile,string, indexer, time, progressbar,csv
from scipy import spatial
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize,word_tokenize
from signatures import *

def check():
	embedding_list = dw.query(indexer.INDEXER_URL, 'SELECT * FROM ' + indexer.INDEX_FILE)
	df = embedding_list.dataframe
	cols  = df.columns.tolist()
	title = ["embedding_name" , "dataset_name", "contributor"]
	return df [title + [field for field in cols if field not in title]]

# SQL query for specific word embedding in given table
def query(words, emb_name, set_name = None, case_sensitive = False):
	set_name = error_check(emb_name, set_name)
	if case_sensitive: first_col = ['column_a', 'the']
	else:
		words = [word.lower() for word in words]
		first_col = ['lower(column_a)', 'lower(the)']

	for title in first_col:
		try:
			if len(words)>1:
				query = 'SELECT * FROM ' + emb_name + ' where ' + title + ' in' + str(tuple(words))
			else:
				query = 'SELECT * FROM ' + emb_name + ' where ' + title + ' = "' + words[0] + '"'
			results = dw.query(set_name, query).dataframe
			results = results[results.columns[::-1]]
			return results.values.tolist()
		except: pass

	raise RuntimeError("No matching word vector found.")

# Get the subset of the pretrained embeddings according to the raw text input.
def extract(emb_name, file_dir='Test_Input/reutersR8_all', set_name= None,download = True):
	set_name = error_check(emb_name, set_name)
	texts = ''
	for name in sorted(os.listdir(file_dir)):
		path = os.path.join(file_dir, name)
		if os.path.isdir(path):
			for fname in sorted(os.listdir(path)):
				fpath = os.path.join(path, fname)
				if sys.version_info < (3,): f = open(fpath)
				else: f = open(fpath, encoding='latin-1')
				texts = texts + f.read()
	input_txt = []
	sentences = sent_tokenize(texts)
	for s in sentences:
		tokens = word_tokenize(s)
		input_txt = input_txt + tokens

	inp_vocab = list(set(input_txt))
	inp_vsize = (len(inp_vocab))

	query, extract_emb = '', []
	print 'Embedding extraction begins.'
	i,loss,title = 0,0,''

	#Extraction is able to recover from Runtime error by adding restore mechanism.
	with progressbar.ProgressBar(max_value=inp_vsize) as bar:
		while i < 400:
			if title != 'column_a':
				try:
					title = 'the'
					query = 'SELECT * FROM ' + emb_name + ' where ' + title + ' in' + str(tuple(inp_vocab[i:i+400]))
					word_vecs = dw.query(set_name, query).dataframe
				except:
					title ='column_a'
					query = 'SELECT * FROM ' + emb_name + ' where ' + title + ' in' + str(tuple(inp_vocab[i:i+400]))
					word_vecs = dw.query(set_name, query).dataframe
			else:
				title ='column_a'
				query = 'SELECT * FROM ' + emb_name + ' where ' + title + ' in' + str(tuple(inp_vocab[i:i+400]))
				word_vecs = dw.query(set_name, query).dataframe

			word_vecs = word_vecs[word_vecs.columns[::-1]].values.tolist()
			loss   += 400 - len(word_vecs)
			extract_emb.extend(word_vecs)
			bar.update(i)
			i += 400

	print 'Embedding successfully extracted.'

	if download == True:
		with open(emb_name+'_extracted.csv', 'w') as extract_csv:
			writer = csv.writer(extract_csv)
			writer.writerows(extract_emb)

	print 'There are ' + str(loss) + " tokens that are in this pretrained word embedding."
	print str(loss/inp_vsize) + "%" + " of words in target corpus are in this pretrained word embedding."
	return extract_emb

# def download(emb_name, set_name=None):
# 	set_name = error_check(emb_name, set_name)
# 	vocab_size = dw.query(indexer.INDEXER_URL, 'SELECT vocab_size FROM ' + indexer.INDEX_FILE + ' WHERE embedding_name = "' + emb_name + '"').dataframe.iloc[0].values[0]
# 	title,i = '',0
# 	with progressbar.ProgressBar(max_value=vocab_size) as bar:
# 		while i < vocab_size:
# 			if title != 'column_a':
# 				try:
# 					title = 'the'
# 					query = 'SELECT * FROM ' + emb_name + ' where ' + title + ' in' + str(tuple(inp_vocab[i:i+400]))
# 					word_vecs = dw.query(set_name, query).dataframe
# 				except:
# 					title ='column_a'
# 					query = 'SELECT * FROM ' + emb_name + ' where ' + title + ' in' + str(tuple(inp_vocab[i:i+400]))
# 					word_vecs = dw.query(set_name, query).dataframe
# 			else:
# 				title ='column_a'
# 				query = 'SELECT * FROM ' + emb_name + ' where ' + title + ' in' + str(tuple(inp_vocab[i:i+400]))
# 				word_vecs = dw.query(set_name, query).dataframe
#
# 			word_vecs = word_vecs[word_vecs.columns[::-1]].values.tolist()
# 			loss   += 400 - len(word_vecs)
# 			download_emb.extend(word_vecs)
# 			bar.update(i)
# 			i += 400
#
# 	if download == True:
# 		with open(emb_name+'.csv', 'w') as download_csv:
# 			writer = csv.writer(download_csv)
# 			writer.writerows(download_emb)
# 	return download_emb

def error_check(emb_name, set_name):
	emb_list = dw.query(indexer.INDEXER_URL, 'SELECT * FROM ' + indexer.INDEX_FILE).dataframe
	emb_names = emb_list.embedding_name

	if len(emb_list.loc[emb_list['embedding_name'] == emb_name]) > 1 and set_name == None:
		raise ValueError("More than one embedding exists with name: " + emb_name + "Try again specifying set_name.")
	if emb_name not in emb_names.values:
		raise ValueError("No embedding exists with name: " + emb_name)
	if emb_list.file_format[emb_names == emb_name].iloc[0] not in ['csv', 'tsv', 'xls', 'rdf', 'json']:
		raise TypeError(emb_name + " was not uploaded in a queryable format. Try reuploading as: csv, tsv, xls, rdf, or json.")

	if not set_name:
		set_name = emb_list.loc[emb_list['embedding_name'] == emb_name].dataset_name.iloc[0]
	return set_name
