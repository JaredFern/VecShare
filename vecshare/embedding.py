import numpy as np
import pandas as pd
import zipfile, os, requests, urllib

class embedding:
	# Initiate the embedding class and check if the embedding we want exists
	def __init__(self,name=None,dimension=None):
		# Load the whole list of current availiable embeddings
		embedding_list = pd.read_csv(indexer.broker_url)
		embedding_names = embedding_list['embedding_name']
		embedding_sizes = embedding_list['vocabulary size']
		embedding_dimensions = embedding_list['dimension']
		embedding_score = 0
		if name == None:
			if len(embedding_list):
				print 'Embeddings now avaliable.'
				print embedding_list
			else:
				print "No embedding is avaliable now."

		self.name = name
		self.dimension = dimension
		self.flag = True

		if name in embedding_names.values:
			url = embedding_list[embedding_names == name]['url'].values[0]
			print 'The embedding you are looking for exists. The url is',url
			self.url = url

			if embedding_list[embedding_names == name]['dimension'].values[0] != dimension:
				print "But the dimension you asked for does not exist."
				self.flag = False

			elif type(embedding_list[embedding_names == name]['dimension'].values[0]) == str:
				if len(embedding_list[embedding_names == name]['dimension'].values[0].split('_')) == 1:
					if embedding_list[embedding_names == name]['dimension'].values[0].split('_')[0] != str(dimension):
						print "But the dimension you asked for does not exist."
						self.flag = False
				else:
					dimension_pool = embedding_list[embedding_names == name]['dimension'].values[0].split('_')
					if str(dimension) not in dimension_pool:
						print "But the dimension you asked for does not exist."
						self.flag = False

		else:
			print 'The embedding you are looking for does not exist.'
			self.flag = False

		if self.flag:
			try:
				self.size = int(embedding_list[embedding_names == name]['vocabulary size'].values[0])
			except:
				if embedding_list[embedding_names == name]['vocabulary size'].values[0][-1] == 'K':
					num = embedding_list[embedding_names == name]['vocabulary size'].values[0][:-1]
					num = int(num) * 1000
					embedding_list[embedding_names == name]['vocabulary size'].values[0] = num
				else:
					num = embedding_list[embedding_names == name]['vocabulary size'].values[0][:-1]
					num = int(num) * 1000000
					embedding_list[embedding_names == name]['vocabulary size'].values[0] = num
			self.path = ''
			self.dl = False
			self.destination = None
			self.vector = None
			self.embed = None
			self.table = embedding_list['table'][embedding_names == name]

	# Download the embeddings in the broker file on data.world and save them
	# on the local files.
	def download(self,path = '_',file_format = 'csv'):
		if len(file_format) > 5:
			print "File format error. Please check it again."
			return
		self.path = path
		if self.flag:
			url = self.url
			path = self.path
			if file_format == 'zip':
				name = url.split('/')[-1]
				if not os.path.exists(path) and path != '_':
					os.makedirs(path)

				urllib.urlretrieve(url,path + name,reporthook = report)
				self.destination = path + name
				print self.destination
				print 'The embedding path is %s .' % os.path.join(os.getcwd(),self.destination)
			else:
				if file_format == 'txt':
					if not os.path.exists(path) and path != '_':
						os.makedirs(path)
					r = requests.get(url,stream = True)
					self.destination = path + self.name + '.txt'
					with open(self.destination,'wb') as f:
						for chunk in r.iter_content(chunk_size = 1024):
							if chunk:
								f.write(chunk)
					print 'The embedding path is %s .' % os.path.join(os.getcwd(),self.destination)
				elif file_format == 'csv':
					if not os.path.exists(path) and path != '_':
						self.path = str(os.getcwd()) + '/' + path
						os.makedirs(self.path)
					elif path == '_':
						self.path = str(os.getcwd()) + '/'
					self.destination = path + self.name + '.csv'
					df = pd.read_csv(self.url)
					df.to_csv(self.destination,index = False)
					print 'The embedding path is %s .' % os.path.join(os.getcwd(),self.destination)
			self.dl = True
		else:
			print "You can't download the embedding because errors happened."
			self.dl = False
		embed = None
		#Extract embedding from zip file or txt file.
		if self.dl:
			if zipfile.is_zipfile(self.destination):
				zf = zipfile.ZipFile(self.destination,'r')
				names = zf.namelist()
				if len(names) != 1:
					dimension = self.dimension
					for filename in names:
						try:
							data = zf.read(filename)
							dimension_of_embed = data.split('\n')[1].split(' ')
							if len(dimension_of_embed) == dimension + 1:
								embed = data
						except KeyError:
							print 'ERROR: Did not find %s in zip file' % filename
				else:
					embed = zf.read(names[0])
			else:
				file = open(self.destination)
				embed = file.read()
			self.embed = embed
			#store the word vector into dictionary
			word_vector = {}
			cach = embed.split('\n')
			for num,row in enumerate(cach):
				if file_format == 'csv':
					values = row.split(',')
				else:
					values = row.split()
				if len(values) < 3:
					continue
				try:
					word = values[0]
					coefs = numpy.asarray(values[1:],dtype = 'float32')
					word_vector[word] = coefs
				except:
					continue
			self.vector = word_vector
			print 'Word embedding has been successfully downloaded.'
			print 'Check the vector attribute by using embedding_name.vector.'
			return word_vector
		else:
			print "The embedding you asked can't be downloaded."
