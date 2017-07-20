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
	set_name = error_check(emb_name, set_name)
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
def extract(file_dir,emb_name, set_name= None,download = False):
	set_name = error_check(emb_name, set_name)

	# Generate test corpus size and vocab
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

	print 'Embedding extraction begins.'
	#Extraction is able to recover from Runtime error by adding restore mechanism.
	query = 'SELECT * FROM ' + emb_name + ' where ' + title + ' in' + str(tuple(words))
	results = dw.query(set_name, query).dataframe
	results = results[results.columns[::-1]]
	results = results.values.tolist()
	if download:
		with open(emb_name+"_extracted.csv", "w") as f:
		    writer = csv.writer(f)
		    writer.writerows(results)
	return results

def download(emb_name, set_name):
	set_name = error_check(emb_name, set_name)
	query = "SELECT * FROM " + emb_name
	results = dw.query(set_name, query).dataframe
	results = results[results.columns[::-1]]
	results = results.values.tolist()
	with open(emb_name+".csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(results)

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
