import pandas as pd
import numpy as np
import datadotworld as dw
import sys,io,os,progressbar,csv,requests
from functools import partial
from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.decomposition import PCA
import pathos.pools as pp

'''
Query: Partial map each group by to a query, rejoin tables (Expand query limit)
Extraction: Same ^^^
Download: Parallel download and join all tables

Signatures: Redirect to operate on appx0
indexer: hotfix + better error handling
'''
try:
	import info, signatures
except ImportError:
	import vecshare.info as info, vecshare.signatures as signatures

def _error_check(emb_name, set_name=None):
	if set_name: return set_name
	emb_list = dw.query(info.INDEXER_URL, 'SELECT * FROM ' + info.INDEX_FILE).dataframe
	emb_names = emb_list.embedding_name

	if len(emb_list.loc[emb_list['embedding_name'] == emb_name]) > 1 and set_name == None:
		raise ValueError("More than one embedding exists with name: " + emb_name + "Try again specifying set_name.")
	if emb_name not in emb_names.values:
		raise ValueError("No embedding exists with name: " + emb_name)
	if emb_list.file_format[emb_names == emb_name].iloc[0] not in ['csv', 'tsv', 'xls', 'rdf', 'json']:
		raise TypeError(emb_name + " was not uploaded in a queryable format. Try reuploading as: csv, tsv, xls, rdf, or json.")

	set_name = emb_list.loc[emb_list['embedding_name'] == emb_name].dataset_name.iloc[0]
	return set_name

def check():
	"""Displays indexed word embeddings and associated metadata.

	Args: NONE
	Returns:
		Pandas Dataframe containing avaiable word embeddings and metadata
	"""
	embedding_list = dw.query(info.INDEXER_URL, 'SELECT * FROM ' + info.INDEX_FILE)
	df = embedding_list.dataframe
	cols  = df.columns.tolist()
	title = ["embedding_name" , "dataset_name", "contributor"]
	return df [title + [field for field in cols if field not in title]]

def format(emb_path,vocab_size=None,dim=None, pca = False, precision=None,sep=","):
	'''Local embeddings will be formatted for upload to the data store as needed:
		* A header will be prepended to the file (text, d1, d2, ..., dn)
		* Elements will be delimited with ","
		* Prefix line from plain text word2vec format:
			Remove "<vocab_size> <dimensionality>"

	Large embeddings can be compressed by reducing the vocab size, dimensionality, or
	precision. `vocab_size` most frequent words will be preserved with `dim` dimensions.
	Embedding values will be limited to `precision` digits. Embedding values will not be
	modified if no optional parameters are specified. If pca flag is set, preserved dimensions
	will be selected using sklearn-PCA. Otherwise the first `dim`+1 columns will be retained.

	Args:
		emb_path(str): Path to embedding
		vocab_size(int,opt): Number of words being retained
		dim(int,opt): Number of dimensions being retained
		pca(bool,opt): Flag for dimension retained
		precision(int,opt): Precision of word vector elements

	Return:
		Modified file for reduced embedding
	'''
	with io.open(emb_path, 'r', encoding='utf-8') as first_pass:
		f_read = csv.reader(first_pass, delimiter =sep)
		first_line = f_read.next()
		header = [u'text']
		header.extend([u"d"+str(n) for n in range(0,len(first_line))])
		if len(first_line) == 2 or first_line == header:
			first_line = f_read.next()
		emb_arr = []
		# If no vocab_size specified, format entire embedding
		if vocab_size:
			vocab_size = vocab_size - 1
			for n in range(0, vocab_size):
				next_line = f_read.next()
				emb_arr.append(next_line)
		else:
			for row in f_read: emb_arr.append(row)

	emb_arr = np.array(emb_arr)

	text = np.array([word if ("#" not in word) or (word[0] == '"' and word[-1] =='"')  else '"' + word + '"' for word in emb_arr[:,0]])
	print ("Reduced vocab_size to: " + str(len(text)))
	wordvecs =emb_arr[:,1:]
	if dim:
		print ("Fitting embedding to lower dimension: " + str(dim))
		if pca:
			wordvecs = wordvecs.astype(float)
			pca = PCA(n_components=dim)
			pca = pca.fit(wordvecs)
			wordvecs = pca.transform(wordvecs)
			wordvecs = wordvecs.astype(str)
		else:
			wordvecs = wordvecs[:,0:dim]
	if precision:
		print ("Reducing precision of vector elements")
		for i in np.nditer(wordvecs, op_flags=['readwrite']):
			i[...] = i[()][0:precision]
	new_emb = np.hstack((text[:,np.newaxis], wordvecs))
	new_emb = new_emb.tolist()
	with io.open(emb_path,'w') as emb_mod:
		print ("Writing modified embedding.")
		write = csv.writer(emb_mod)
		write.writerow(header)
		for each in new_emb:
			write.writerow(each)

def upload(set_name, emb_path="", metadata = {}, summary = None):
	'''Upload a new embedding or update files and associated metadata.

	Args:
		set_name (str): Name of the dataset being created (format: owner/id)
		emb_path (str): Absolute path to local embedding
		metadata (dict, opt): Dictionary in the format '{metadata field: value}'
		summary (str, opt): Optional description of embedding and source

	Returns: None (Create a new/updated data.world dataset with the shared embedding)
	'''
	dw_api = dw.api_client()
	set_name = set_name.replace(' ', '-').replace('_','-')
	metadata_str, dimensions, app_num = "", 0, 0
	usr_name, title = set_name.split("/")
	emb_name = os.path.basename(emb_path)

	for key,val in metadata.items():
		metadata_str += str(key) + ":" + str(val) + ", "

	with io.open(emb_path, 'r', encoding='utf-8' ) as f: first_row = f.readline().split(',')
	header  = ['text']
	header.extend([u"d"+str(n) for n in range(len(first_row)-1)])

	emb_reader = pd.read_csv(emb_path, chunksize = 200000, names = header,encoding ='utf-8')
	index_df = pd.DataFrame()
	for app_num, emb_chunk in enumerate(emb_reader):
		app_title 	= title+"-appx" + str(app_num)
		app_setname = usr_name+"/"+app_title
		app_fname 	= app_title + ".csv"

		words = emb_chunk.ix[:,0].reset_index(drop=True)
		app_sets = pd.Series(app_setname, index= np.arange(len(emb_chunk)), name="app_setname")
		app_file = pd.Series(app_fname, index=np.arange(len(emb_chunk)), name="app_fname")

		tmp_df = pd.concat((words, app_sets ,app_file), axis=1,copy=False)
		index_df =index_df.append(tmp_df,ignore_index=True)
		try:
			dw_api.create_dataset(usr_name, title = app_title, description = summary,\
            license = 'Public Domain', tags = ['vecshare appx'], visibility = 'OPEN')
		except:
			dw_api.update_dataset(app_setname, description=summary)
		with dw.open_remote_file(app_setname, app_fname, 'w') as app:
			emb_chunk.to_csv(app, encoding='utf-8',index = False)
	try:
		metadata_str += "app_num:"+str(app_num+1)
		dw_api.create_dataset(usr_name, title = title, summary = metadata_str, description = summary,\
		license = 'Public Domain', tags = ['vecshare large'], visibility = 'OPEN')
	except:
		dw_api.update_dataset(set_name, summary = metadata_str, description=summary)
	with dw.open_remote_file(set_name, emb_name,'w') as index:
		index_df.to_csv(index, encoding='utf-8', index =False)

def query(words, emb_name, set_name = None, case_sensitive = False,download=False):
	"""Query a set of word vectors from an indexed embedding.
	Args:
		words (List of strings): The set of word vectors being queried
		emb_name (str): The embedding from which word vectors are being queried
		set_name (str, opt): Name of dataset being queried (format: owner/id)
		case_sensitive (bool, opt): Flag for matching exact case in query
		download(bool,opt): Flag for saving query results
	Returns:
		Pandas Dataframe, each row specifying a word vector.
	 """
	 # Set names should have '-' and file names should use "_"
	def partial_query(args):
	    return dw.query(args[0], args[1]).dataframe

	if case_sensitive: title = 'text'
	else:
		words = [word.lower() for word in words]
		title = 'lower(text)'
	set_name = _error_check(emb_name, set_name)
	query_list,proc_cnt=[],16
	ind_results, combined_vecs = pd.DataFrame(),pd.DataFrame()
	multiproc = pp.ProcessPool(proc_cnt)

	try:
		ind_query = 'SELECT * FROM ' + emb_name + ' where ' + title
		if len(words)==1:
			cond = '"' + words[0] + '"'
			ind_results = dw.query(set_name, ind_query + cond).dataframe
		else:
			for i in range(0,len(words), 400):
				query_words=words[i:i+400]
				query_list.append([set_name, ind_query + ' in'+ str(tuple(query_words)) ])
			word_index = multiproc.map(partial_query, query_list)
			word_index = [word for word in word_index]
			for each in word_index:
				ind_results= ind_results.append(each)
	except:
 		RuntimeError("Embedding is formatted improperly. Check upload at: " + set_name)

	num_appx = "SELECT app_num FROM " +info.INDEX_FILE + " WHERE embedding_name = " + emb_name +" and dataset_name = " + set_name
	app_count = dw.query(info.INDEXER, num_appx).dataframe.iloc[0][0]
	query_list=[]
	for each in range(app_count):
		base_query = 'SELECT * FROM ' + emb_name.lower() + "_appx" + str(each)+ ' where ' + title
		ind_appcnt = (ind_results[(ind_results['app_setname'] == set_name + "-appx" + str(each))])['text']
		for i in range(0, len(ind_appcnt), 400):
			if len(words)>1: 	cond = ' in' + str(tuple(words[i:i+400]))
			else: 				cond = ' = "' + words[0] + '"'
			query_list.append([set_name + "-appx" + str(each), base_query + cond])
	word_vecs = multiproc.map(partial_query,query_list)
	for each in word_vecs:
		combined_vecs =combined_vecs.append(each)
	if download == True:
		with io.open(emb_name+'_query.csv', 'w', encoding='utf-8') as extract_csv:
			extract_emb.to_csv(extract_csv, encoding = 'utf-8', index = False)
	multiproc.terminate()
	return combined_vecs

def extract(emb_name, file_dir, set_name = None, case_sensitive = False, download = False, progress=True):
	"""Queries word vectors from `emb_name` for all words in the target corpus.
	Args:
		emb_name(str): Name of the selected embedding
		file_dir(PATH): Path to the target corpus
		set_name(opt, str): Specify if multiple embeddings exist with same name
		download(bool): Flag for saving the extracted embeddingas csv

	Returns:
		Pandas DataFrame containing queried word vectors
	"""
	set_name = _error_check(emb_name, set_name)
	inp_vocab = set()

	for root,dirs,files in os.walk(file_dir):
		files = [f for f in files if f[0]!= '.']
		for fname in files:
			fpath = os.path.join(root, fname)
			f = io.open(fpath, 'r', encoding = 'utf-8')
			# if sys.version_info < (3,):
			# 	f = open(fpath)
			# else:
			# 	f = open(fpath, encoding='utf-8')
			sentences = sent_tokenize(f.read())
			for s in sentences:
				inp_vocab.update(word_tokenize(s))

	if case_sensitive: inp_vocab = list(inp_vocab)
	else: inp_vocab = [word.lower() for word in list(inp_vocab)]
	inp_vsize = len(inp_vocab)
	print ('Embedding extraction begins.')
	extract_emb = query(inp_vocab, emb_name, set_name)
	print ('Embedding successfully extracted.')
	if download == True:
		with io.open(emb_name+'_extracted.csv', 'w', encoding='utf-8') as extract_csv:
			extract_emb.to_csv(extract_csv, encoding = 'utf-8', index = False)
	return extract_emb

def download(emb_name, set_name=None):
	'''Loads the full shared embedding `emb_name` and saves the embedding to the
	current working directory.

	Args:
		emb_name(str): Name of the selected embedding
		set_name(opt, str): Specify if multiple embeddings exist with same name

        Returns:
		.csv embedding saved to the current working directory
	'''
	DW_API_TOKEN = os.environ['DW_AUTH_TOKEN']
	multiproc = pp.ProcessPool(proc_cnt)
	def emb_download(appx_num) :
		query_url = "https://query.data.world/file_download/"+set_name+"-appx" + str(appx_num)+"/"+ emb_name+ "-appx" + str(appx_num)  + '.csv'
		payload, headers = "{}", {'authorization': 'Bearer '+ DW_API_TOKEN}
		emb_text = requests.request("GET", query_url, data=payload, headers=headers).text
		with io.open(emb_name+"-appx"+str(appx_num) + '.csv', 'w', encoding='utf-8') as download_emb:
			download_emb.write(emb_text)

	set_name = _error_check(emb_name)
	num_appx = "SELECT app_num FROM " +info.INDEX_FILE + " WHERE embedding_name = " + emb_name +" and dataset_name = " + set_name
	app_count = dw.query(info.INDEXER, num_appx).dataframe.iloc[0][0]
	multiproc.map(emb_download, list(range(num_appx)))

	with io.open(emb_name+'.csv', 'w', encoding='utf-8') as compiled:
		first_appx = io.open(emb_name+'-appx0.csv', 'r', encoding = 'utf-8')
		compiled.write(first_appx.read())
		for i in range(1,app_count):
			with io.open(emb_name+"-appx"+str(i) + '.csv', 'r', encoding = 'utf-8') as appx:
				appx.next()
				for line in appx: compiled.write(line)
			os.remove(emb_name+"-appx"+str(i) + '.csv')
	return pd.read_csv(emb_name+'.csv')
