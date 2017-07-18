import Embedding_platform as ep

#Check the information of all avaiable embeddings existing in the broker.csv
ep.check()

# Query for a specified word in specfied embedding.
print ep.query_embeddings('arts_40','you')
print ep.query_embeddings('agriculture_40','the')		

# Extracted necessary parts of pre_trained embeddings for given corpus input. 
# 'pad' = True by default in order to add zeros to fill up our embedding for missing words.
# 'check' = True by default in order to display embeddings we extacted.
ep.EmbedExtract(file_dir ='Test_Input/reutersR8_all' ,table ='art_40',pad = True,check = True)
ep.EmbedExtract(file_dir ='Test_Input/reutersR8_all' ,table ='arts_40',pad = True,check = True,download = True)

# Initiate word embedding by specifying its name and dimension.
# Initiation will fail if wrong dimension parameter given or embedding doesn't exist.
a = ep.embedding('agriculture_40',100)
print a.name
print a.dimension
a.download(path = 'Embedding/')


INPUT_DIR = 'Test_Input/reutersR8_all'

# Embedding selection by method A.
ep.method_a(inp_dir = INPUT_DIR,num_sig = 5000,num_sig_embedding = 5000,num_stopwords = 100)
