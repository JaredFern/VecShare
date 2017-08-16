import pandas as pd
import numpy as np
import datadotworld as dw
import sys,os,progressbar,csv,requests
from functools import partial
from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.decomposition import PCA
from multiprocessing import Pool

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
		pca(bool,opt): Flag for dimension red
		precision(int,opt): Precision of word vector elements

	Return:
		Modified file for reduced embedding
	'''
	header = ""
	with open(emb_path, 'r') as first_pass:
		first_line = first_pass.readline()
		header = "text,d" + ",d".join(str(n) for n in range(0,len(first_line.split(sep)))) + "\n"
		if len(first_line.split()) == 2 or first_line == header:
			first_line = first_pass.readline()
		emb_arr = [first_line.strip().split(sep)]
		vocab_size = vocab_size - 1
		# If no vocab_size specified, format entire embedding
		if vocab_size:
			for n in range(0, vocab_size):
				next_line = first_pass.readline().strip()
				emb_arr.append(next_line.split(sep))
		else:
			lines = first_pass.readlines()
			emb_arr.append([x.strip().split(sep) for x in lines])
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
	with open(emb_path,'w') as emb_mod:
		print ("Writing modified embedding.")
		write = csv.writer(emb_mod)
		emb_mod.write(header)
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
	metadata_str = ""
	for key,val in metadata.items():
		metadata_str += str(key) + ":" + str(val) + ", "
	try:
		usr_name, title = set_name.split("/")
		dw_api.create_dataset(usr_name, title = title, summary = metadata_str,\
		description = summary ,license = 'Public Domain', tags = ['vecshare'], visibility = 'OPEN')
	except:
		dw_api.update_dataset(set_name, summary = metadata_str, description=summary)

	if emb_path:
		dw_api.upload_files(set_name, [emb_path])

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
		return results
	except:
		raise RuntimeError("Embedding is formatted improperly. Check headers at: " + query)
	raise RuntimeError("No matching word vector found.")

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
		for f in files:
			fpath = os.path.join(root, f)
			if sys.version_info < (3,): f = open(fpath)
			else: f = open(fpath, encoding='utf-8')
			sentences = sent_tokenize(f.read())
			for s in sentences:
				inp_vocab.update(word_tokenize(s))

	if case_sensitive: inp_vocab = list(inp_vocab)
	else: inp_vocab = [word.lower() for word in list(inp_vocab)]
	inp_vsize = len(inp_vocab)
	print ('Embedding extraction begins.')
	query, extract_emb = '', pd.DataFrame()
	i,loss, proc_cnt, query_size = 0,0,16,400

	if progress == True: bar = progressbar.ProgressBar(max_value=inp_vsize)

	p = Pool(proc_cnt)
	while i < inp_vsize:
		if case_sensitive: case = 'text'
		else: case = 'lower(text)'
		query = ['SELECT * FROM ' + emb_name + ' where '+ case + ' in' +  \
			str(tuple(inp_vocab[i + query_size*j :i + query_size*(j+1)])) \
			for j in range(0,proc_cnt)]
		partial_map = partial(dw.query, set_name)
		word_vecs = p.map(partial_map, query)
		word_vecs = [vec.dataframe for vec in word_vecs]

		for each in word_vecs:
			extract_emb = extract_emb.append(each)

		loss += query_size * proc_cnt - len(word_vecs)
		if progress == True: bar.update(i)
		i += query_size * proc_cnt

	print ('Embedding successfully extracted.')
	if download == True:
		with open(emb_name+'_extracted.csv', 'w') as extract_csv:
			extract_emb.to_csv(extract_csv, encoding = 'utf-8', index = False)
	p.terminate()
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
	set_name = _error_check(emb_name)
	query_url = "https://query.data.world/file_download/"+set_name+"/"+ emb_name + '.csv'
	payload, headers = "{}", {'authorization': 'Bearer '+ DW_API_TOKEN}
	emb_text = requests.request("GET", query_url, data=payload, headers=headers).text

	with open(emb_name + '.csv', 'w') as download_emb:
		download_emb.write(emb_text.encode('utf-8'))

	return pd.read_csv(emb_name+'.csv')
