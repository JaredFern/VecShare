# VecShare: Framework for Sharig Word Embeddings
Library is in the process of being updated. See https://github.com/MarcusYYY/WordEmbeddingPlatform for a stable test version of the VecShare framework. A fully operational release will be publicly available prior to the EMNLP 2017 conference, by **September 7**.

A Python library for word embedding query, selection and download. Read more about VecShare: <PAPER_URL>.

## Prerequisites:
Before installing this library, install the datadotworld Python library:
```
pip install git+git://github.com/datadotworld/data.world-py.git
```

Configure the datadotworld library with your data.world API token.
Your token is obtainable on data.world under [Settings > Advanced](https://data.world/settings/advanced)

Set your token:
```
export DW_AUTH_TOKEN=<YOUR_TOKEN>   \\ Mac OSX or Unix Systems

dw configure                        \\ Windows Systems
```

## Quickstart:
Install the VecShare Python library:
```
pip install vecshare
```

See **Advanced Setup** for details on creating new indexers or signature methods.

### Check Available Embeddings
The `check()` method displays embeddings available with the current indexer. The default indexer aggregates a set of embeddings by polling `data.world` weekly for datasets with the tag `vecshare`. Indexed embeddings are viewable at: `https://data.world/jaredfern/vecshare-indexer`.

See **Advanced Setup**, if you would like to use a custom indexer.

For example:
```python
>>>import vecshare
>>>vecshare.check()
embedding_name                                        agriculture_40
dimension                                                        100
vocabulary size                                                19007
url                https://query.data.world/s/ewf2464mqcr7tyiatbh...
table                                                 agriculture_40
file_format                                                      csv
```

### Embedding Query
The `query(table, request)` method returns word vectors for the words in array `request` from shared embedding `table`.

For example:
```python
>>>print vecshare.query('agriculture_40',['the'])
['the', -1.004704, 0.037287, -0.016309, -0.088428, -1.1478, 0.331032, -0.77213, -0.07757, -0.874058, -1.170626, -0.253766, 1.137803, 1.045363,
 2.386086, 0.229137, 0.272712, -0.334886, -1.015797, 0.662011, -0.472902, -0.333736, 1.604692, 0.924259, 0.707687, -0.153192, 1.007494, 1.09558,
-1.159106, 0.88615, 1.214197, -1.345269, -2.309988, 0.581767, -2.040186, 0.019013, -0.090971, -0.690396, 1.578381, -0.441838, 0.968358, 0.865741,
-1.263163, -0.829032, -0.313665, 0.138191]
```
### Embedding Upload
New embeddings can be added to the framework by uploading the embedding as a .csv file to data.world, and tagging the dataset with the vecshare tag. The default indexer will add new embedding sets weekly.

Metadata associated with the embedding can be added in the datasets description in the following format, `Field: Value`

For Example:
```
Embedding Type: word2vec
Token Count: 6000000
Case Sensitive: False
```

### Embedding Extraction
The `extract(file_path, table, download=False)` method return vectors for the words in the shared embedding `table` that overlap with the user's corpus at `corp_path`.

Parameters:
* **corp_path (str):**  Absolute path to the directory containing the target corpus
* **table (str):**      Name of the embedding on the framework to be downloaded
* **padding (bool, Optional):** If padding flag is set to True, word vectors with value-zero will be added for words in the target corpus but not in the embedding


For example:
```
>>> vecshare.extract(file_dir ='Test_Input/reutersR8_all' ,table ='agriculture_40',pad = True,check = True,download = True)
```

### Embedding Download

For example:
```
>>> A = ep.embedding('agriculture_40',100)
The embedding you are looking for exists. The url is https://query.data.world/s/enfkzx0yrnxevzcy9m7fm81hi
>>> print A.name
agriculture_40
>>> print A.dimension
100
>>>A.download(path = 'Embedding/')
The embedding path is /Users/Embedding_platform/Embedding/agriculture_40.csv .
Word embedding has been successfully downloaded.
Check the vector attribute by using embedding_name.vector
```
### Embedding Selection
The `AvgRank(inp_dir,num_sig,num_sig_embedding,num_stopwords)` embedding selection method that selects the embedding with highest similarity

We are assuming that word embeddings with similar high frequency words are built from similar topics corpor. The score this method computed is negatively correlated with the performance of those pretrained word emebddings.Therefore the emebdding with the least score outperforms the others.

`inp_dir` is the path of input corpora.`num_sig` means the number of words chosen as signature of the input corpora. `num_sig_embedding` indicates the number of words picked as signature of the pretrained embedding. And`num_stopwords` is the number of stopwords we ignore in this method.

For example:
```python
>>>INPUT_DIR = 'Test_Input/reutersR8_all'
>>>ep.method_a(inp_dir = INPUT_DIR,num_sig = 5000,num_sig_embedding = 5000,num_stopwords = 100)
100 most frequent words will be removed as stop words.
Pick up 5000 top frequent words as signature of all avaliable embeddings.
Fetching agriculture_40 embeddings and creating its signature.
Fetching art_40 embeddings and creating its signature.
Fetching books_40 embeddings and creating its signature.
Fetching econ_40 embeddings and creating its signature.
Fetching govt_40 embeddings and creating its signature.
Fetching movies_40 embeddings and creating its signature.
Fetching weather_40 embeddings and creating its signature.
Processing High Density Vocabulary list.
Assembled High Density Vocabulary found in at least  85.0  percent of embeddings.
Pick top 5000 most frequently occurring words in the corpus as signature.
[('movies_40', 2892.3935935935065),
 ('books_40', 2915.7785785784804),
 ('art_40', 3016.7237237236086),
 ('agriculture_40', 3236.933733733591),
 ('govt_40', 3243.128928928771),
 ('weather_40', 3267.9365365363774),
 ('econ_40', 3275.4096096094536)]
 The best pretrained embedding is movies_40.
 ```

## Advanced Setup
### Custom Signature Methods:


### Custom Indexers:
