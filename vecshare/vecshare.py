import pandas as pd
import numpy as np
import datadotworld as dw
import sys,os,progressbar,csv,indexer,signatures, pdb
from functools import partial
from nltk.tokenize import sent_tokenize,word_tokenize
from multiprocessing import Pool

def _error_check(emb_name, set_name):
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

def check():
	"""Displays indexed word embeddings and associated metadata.

	Args: NONE
	Returns:
		Pandas Dataframe containing avaiable word embeddings and metadata
	"""
	embedding_list = dw.query(indexer.INDEXER_URL, 'SELECT * FROM ' + indexer.INDEX_FILE)
	df = embedding_list.dataframe
	cols  = df.columns.tolist()
	title = ["embedding_name" , "dataset_name", "contributor"]
	return df [title + [field for field in cols if field not in title]]

def format(emb_path):
	"""Adds a header to an embedding in .csv format to support tabular access.
	Args:
		emb_path(filepath): File path pointing at taret embedding
	Returns:
		None (Reformatted .csv embedding)
	"""
	data, row_count = "", 0
	with open (emb_path, 'r') as emb:
		csv_reader = csv.reader(emb)
		row_count = len(csv_reader.next())
		emb.seek(0)
		data = emb.read()
	with open (emb_path, 'w') as emb:
		header = "text,d" + ",d".join(str(n) for n in range(0,row_count)) + "\n"
		emb.write(header)
		emb.write(data)

def query(words, emb_name, set_name = None, case_sensitive = False):
	"""Query a set of word vectors from an indexed embedding.
	Args:
		words (List of strings): The set of word vectors being queried
		emb_name (str): The embedding from which word vectors are being queried
		set_name (str, opt): Name of dataset being queried (format: owner/id)
		case_sensitive (bool, opt): Flag for matching exact case in query

	Returns:
		Pandas Dataframe, each row specifying a word vector.
	 """
	set_name = _error_check(emb_name, set_name)
	if case_sensitive: title = 'text'
	else:
		words = [word.lower() for word in words]
		title = 'lower(text)'
		try:
			if len(words)>1:
				query = 'SELECT * FROM ' + emb_name + ' where ' + title + ' in' + str(tuple(words))
			else:
				query = 'SELECT * FROM ' + emb_name + ' where ' + title + ' = "' + words[0] + '"'
			results = dw.query(set_name, query).dataframe
			results = results[results.columns[::-1]]
			return results
		except:
			raise RuntimeError("Embedding is formatted improperly. Check headers at: " + query)
	raise RuntimeError("No matching word vector found.")

def extract(emb_name, file_dir, set_name = None, case_sensitive = False, download = False):
	"""Queries word vectors from selected embedding for all words in the target corpus.
	Args:
		emb_name(str): Name of the selected embedding
		file_dir(PATH): Path to the target corpus
		set_name(opt, str): If multiple embeddings exist with same name, specify the datset
		download(bool): Flag for saving the extracted embeddingas csv

	Returns:
		Pandas DataFrame containing queried word vectors
	"""
	set_name = _error_check(emb_name, set_name)
	inp_vocab = set()

	for name in sorted(os.listdir(file_dir)):
		path = os.path.join(file_dir, name)
		if os.path.isdir(path):
			for fname in sorted(os.listdir(path)):
				fpath = os.path.join(path, fname)
				if sys.version_info < (3,): f = open(fpath)
				else: f = open(fpath, encoding='utf-8')

				sentences = sent_tokenize(f.read())
				for s in sentences:
					inp_vocab.update(word_tokenize(s))
	if case_sensitive: inp_vocab = list(inp_vocab)
	else: inp_vocab = [lower(word) for word in list(inp_vocab)]
	inp_vsize = len(inp_vocab)

	print ('Embedding extraction begins.')
	query, extract_emb = '', pd.DataFrame()
	i,loss, proc_cnt, query_size = 0,0,16,400

	with progressbar.ProgressBar(max_value=inp_vsize) as bar:
		p = Pool(proc_cnt)
		while i < inp_vsize:
			if case_sensitive: case = 'text'
			else: case = 'lower(text)'
			query = ['SELECT * FROM ' + emb_name + ' where '+ case + ' in' + \
				str(tuple(inp_vocab[i + query_size*j :i + query_size*(j+1)])) for j in range(0,proc_cnt)]
			partial_map = partial(dw.query, set_name)
			word_vecs = p.map(partial_map, query)
			word_vecs = [vec.dataframe for vec in word_vecs]

			for each in word_vecs:
				extract_emb = extract_emb.append(each[each.columns[::-1]])

			loss += query_size * proc_cnt - len(word_vecs)
			bar.update(i)
			i += query_size * proc_cnt

	print ('Embedding successfully extracted.')
	if download == True:
		with open(emb_name+'_extracted.csv', 'w') as extract_csv:
			extract_emb.to_csv(extract_csv, encoding = 'utf-8', index = False)
	return extract_emb

def upload(set_name, emb_path, metadata = {}, summary = None):
	'''Upload an embedding to a new data set on the data.world datastore

	Args:
		usrs_name(str): data.world user name
		set_name (str): Name of the dataset being created (format: owner/id)
		emb_path (str): Absolute path to local embedding
		metadata (dict, opt): Dictionary in the format '{metadata field: value}'
		summary (str, opt): Optional description of embedding and source

	Returns: None (Create a new data.world dataset with the shared embedding)
	'''
	dw_api = dw.api_client()
	metadata_str = ""
	for key,val in metadata.items():
		metadata_str += key + ":" + val + ", "

	usr_name, set_name = ("/").split(set_name)
	dw_api.create_dataset(usr_name, title = set_name, summary = metadata_str, \
		desription = summary ,license = 'Public Domain', tags = 'vecshare', \
		visibility = 'OPEN')

	dw_api.upload_files(set_name, [emb_path])

def update(set_name, emb_path = "", metadata = {}, summary = ""):
	'''Update the embedding or metadata for an existing dataset on data.world

	Args:
		set_name (str): Name of the dataset being updated (format: owner/id)
		emb_path (str): Absolute path to local embedding
		metadata (dict, opt): Dictionary in the format '{metadata field: value}'
		summary (str, opt): Optional description of embedding and source

	Returns: None (Create a new data.world dataset with the shared embedding)
	'''
	dw_api = dw.api_client()
	if emb_path:
		dw_api.upload_files(set_name, emb_path)

	if metadata or emb_description:
		metadata_str = ""
		for key,val in metadata.items():
			metadata_str += key + ":" + val + ", "
		dw_api.update_dataset(set_name, summary = metadata_str, description=summary)

# def download(emb_name, set_name=None):
# 	set_name = _error_check(emb_name, set_name)
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
