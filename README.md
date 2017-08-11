# VecShare: Framework for Sharing Word Embeddings

## About VecShare
The vecshare python library for word embedding query, selection and download. The vecshare python library uses indexers to regularly poll the data.world datastore for uploaded embeddings, record associated metadata, and generate lightweight signatures representing each uploaded embedding. Users can select embeddings for use by specifying the name of the desired embedding or using provided methods to compare their corpus against indexed signatures and extracting the embedding most similar to the target corpus.

Read more about VecShare: <https://bit.ly/VecShare>.

## Embedding Leaderboard
Each indexed is evaluated and assigned a score on 10 word pair similarity tasks. The **score** is calculated by measuring the average Spearman correlation of the word vector cosine similarities and human-rated similarity for each word pair.

**Highest Scoring Word Embeddings:**

[comment]: <> (Leaderboard Start)

| embedding_name                                                                                   | dataset_name                             | contributor   | embedding_type   |   dimension |    score |
|:-------------------------------------------------------------------------------------------------|:-----------------------------------------|:--------------|:-----------------|------------:|---------:|
| [`glove_Gigaword100d`](https://data.world/jaredfern/gigaword-glove-embedding/glove_Gigaword100d) | jaredfern/gigaword-glove-embedding       | jaredfern     | glove            |         100 | 0.456143 |
| [`text8_emb`](https://data.world/jaredfern/text-8-w-2-v/text8_emb)                               | jaredfern/text-8-w-2-v                   | jaredfern     | word2vec         |          50 | 0.37306  |
| [`books_40`](https://data.world/jaredfern/new-york-times-word-embeddings/books_40)               | jaredfern/new-york-times-word-embeddings | jaredfern     | word2vec         |         100 | 0.303337 |
| [`OANC_Written`](https://data.world/jaredfern/oanc-word-embeddings/OANC_Written)                 | jaredfern/oanc-word-embeddings           | jaredfern     | word2vec         |         100 | 0.293891 |
| [`econ_40`](https://data.world/jaredfern/new-york-times-word-embeddings/econ_40)                 | jaredfern/new-york-times-word-embeddings | jaredfern     | word2vec         |         100 | 0.290213 |
| [`agriculture_40`](https://data.world/jaredfern/new-york-times-word-embeddings/agriculture_40)   | jaredfern/new-york-times-word-embeddings | jaredfern     | word2vec         |         100 | 0.289704 |
| [`govt_40`](https://data.world/jaredfern/new-york-times-word-embeddings/govt_40)                 | jaredfern/new-york-times-word-embeddings | jaredfern     | word2vec         |         100 | 0.288382 |
| [`weather_40`](https://data.world/jaredfern/new-york-times-word-embeddings/weather_40)           | jaredfern/new-york-times-word-embeddings | jaredfern     | word2vec         |         100 | 0.277633 |
| [`arts_40`](https://data.world/jaredfern/new-york-times-word-embeddings/arts_40)                 | jaredfern/new-york-times-word-embeddings | jaredfern     | word2vec         |         100 | 0.266848 |
| [`movies_40`](https://data.world/jaredfern/new-york-times-word-embeddings/movies_40)             | jaredfern/new-york-times-word-embeddings | jaredfern     | word2vec         |         100 | 0.263914 |

[comment]: <> (Leaderboard End)

**Word Pair Similarity Tasks:**
* WS-353: Finkelstein et. al, 2002
* MC-30: Miller and Charles, 1991
* MEN: Bruni et. al, 2012
* MTurk-287: Radinsky et. al, 2011
* MTurk-771: Halawi and Dror, 2012
* Rare-Word: Luong et. al, 2013
* SimLex-999: Hill et. al, 2014
* SimVerb-3500: Gerz et. al, 2016
* Verb-144: Baker et. al, 2014

## Installation:
Install the VecShare Python library:
```
pip install vecshare
```
Before using the `vecshare` library, configure the datadotworld library with your API token.
Your token is obtainable on data.world under [Settings > Advanced](https://data.world/settings/advanced)

Set your data.world token using:
```
dw configure
```
or
```
export DW_AUTH_TOKEN=<DATA.WORLD_API_TOKEN>
```
To avoid resetting the token for each terminal instance, add your token as an environment variable to your bash profile or permanent environment variables.

See [**Advanced Setup**](#advanced-setup) for details on creating new indexers or signature methods.

## Supported Functions
The VecShare Python library currently supports:
  * [`check`](#check-available-embeddings): See available embeddings
  * [`format`](#embedding-upload-or-update): Autoformat an embedding for upload to the data store
  * [`upload`](#embedding-upload-or-update): Upload a new embedding to the datastore
  * [`update`](#embedding-upload-or-update): Update an existing embedding or its metadata
  * [`query`](#embedding-query): Look up word vectors from a specific embedding
  * [`extract`:](#embedding-extraction) Download word vectors for only the vocabulary of a specific corpus
  * [`download`](#full-embedding-download): Download an entire shared embedding

### Check Available Embeddings
**`check()`:**  Returns embeddings available with the current indexer as a queryable `pandas.DataFrame`.

The default indexer aggregates a set of embeddings by polling `data.world` weekly for datasets with the tag `vecshare`. Currently indexed embeddings are viewable at: <https://data.world/jaredfern/vecshare-indexer>.

See [**Advanced Setup**](#advanced-setup), if you would like to use a custom indexer.

**For Example:**
```python
>>> from vecshare import vecshare as vs
>>> vs.check()
        embedding_name                              dataset_name contributor  \
0            reutersr8         jaredfern/reuters-word-embeddings   jaredfern   
1         reuters21578         jaredfern/reuters-word-embeddings   jaredfern   
2                brown                    jaredfern/brown-corpus   jaredfern   
3   glove_gigaword100d        jaredfern/gigaword-glove-embedding   jaredfern   
4         oanc_written            jaredfern/oanc-word-embeddings   jaredfern   


   case_sensitive  dimension embedding_type file_format vocab_size
0           False        100       word2vec         csv       7821
1           False        100       word2vec         csv      20203     
2           False        100       word2vec         csv      15062     
3           False        100          glove         csv     399922
4           False        100       word2vec         csv      73127        
```

### Embedding Upload or Update
Embeddings must be uploaded as a .csv file with a header in the format: ['text', 'd0', 'd1', ... 'd_n'], such that they can be properly indexed and accessed.

**`format(emb_path)`:** Reformats existing embeddings in the .csv format with a header in the correct format for tabular access.
  * **emb_path (str):** Path to the embedding being formatted

**`upload(set_name, emb_path, metadata = {}, summary = None)`:** Create a new shared embedding on data.world
  * **set_name (str):** Name of the new dataset on data.world in the form (data.world_username/dataset_name)
  * **emb_path (str):** Path to embedding being uploaded
  * **metadata (dict, opt):** Dictionary containing metadata fields and values '{metadata_field: value}'
  * **summary (str, opt):** Optional embedding description

**`update(set_name, emb_path = "", metadata = {}, summary = "")`:** Update an existing shared embedding or its associated metadata
  * **set_name (str):** Name of the new dataset on data.world in the form (data.world_username/dataset_name)
  * **emb_path (str):** Path to embedding being uploaded
  * **metadata (dict, opt):** Dictionary containing metadata fields and values '{metadata_field: value}'
  * **summary (str, opt):** Optional embedding description

Alternatively, new embeddings can be added to the framework by uploading the embedding as a .csv file to data.world, and tagging the dataset with the <vecshare> tag. The default indexer will add new embedding sets weekly.

Metadata associated with the embedding can be added in the datasets description in the following format, `Field: Value`

**For example:**
```
Embedding Type: word2vec
Token Count: 6000000
Case Sensitive: False
```
### Embedding
**`signatures.avgrank(inp_dir)`:** Returns the shared embedding most similar to the user's target corpus, using the AvgRank method described in the VecShare paper. *Note: Computation is performed locally. Users' corpora will not be shared with other users*
* **inp_dir (str):** Path to the directory containing the target corpus.

```python
>>> from vecshare import signatures as sigs
>>> sigs.avgrank('Test_Input')
u'reutersR8
```
**`signatures.simscore():`** Returns the embedding currently scoring highest on the word pair similarity task suite.

Additional custom  similarity and selection methods can be added. See ['Advanced Setup'](#advanced-setup).
### Embedding Query
**`query(words, emb_name, set_name = None, case_sensitive = False)`:**  Returns a  pandas DataFrame, such that each row specifies a word vector from the query.
  * **words (list):** List of word vectors being requested
  * **emb_name (str):** Title of the embedding containing the requested word vectors
  * **set_name (str, opt):** Specify if multiple embeddings exist with the same emb_name
  * **case_sensitive (bool):** Set to True if word vectors must exactly case match those in words

**For Example:**
```python
>>> from vecshare import vecshare as vs
>>> vs.query(['The', 'farm'], 'agriculture_40')
   text       d99       d98       d97       d96       d95   ...           d1      d0  
0   the -1.414755  0.414973  1.115698  0.034085  0.542921   ...   0.037287 -1.004704  
1  farm  0.349535 -0.379208 -0.189476  2.776809 -0.099886   ...   0.067443 -1.391604  
[2 rows x 101 columns]
```
### Embedding Extraction
**`def extract(emb_name, file_dir, set_name = None, download = False):`** Return a pandas DataFrame containing all available word vectors for the target corpora's vocabulary.

Parameters:
  * **emb_name (str):** Title of the shared embedding
  * **file_dir (str):** Directory containing the user's target corpora
  * **set_name (str,opt):** Specify only if multiple embeddings exist with the same emb_name
  * **download (bool,opt):** If True, the extracted embedding will be saved as a .csv
  * **case_sensitive (bool):** Set to True if word vectors must exactly case match those in words

**For example:**
```python
>>> from vecshare import vecshare as vs
>>> vs.extract('agriculture_40', 'Test_Input/reutersR8_all')
Embedding extraction begins.
100% (23584 of 23584) |################################| Elapsed Time: 0:01:04
Embedding successfully extracted.

              text       d99       d98       d97       d96       d95    ... \
0        designing -0.194328 -0.229856  0.455848  0.234053 -0.272354    ...
1       affiliated -0.446879 -0.519360  0.130626  0.034608  0.134680    ...
2    appropriately  0.106778  0.057186 -0.222296  0.101948  0.395122    ...
3       cincinnati -0.563716 -0.274534  0.120897  0.273457  0.383307    ...
4           choice  0.689276  1.586349  1.301351 -1.193058 -0.243053    ...
5              han -0.287583  0.237989 -0.141203  0.328414  0.401448    ...
6            begin  1.952841 -1.497073 -0.656650  2.443687  0.315941    ...
7        wednesday -1.591453 -1.419733 -0.758305  2.638620  0.323779    ...
8            wales -0.591623 -0.761353 -0.042557 -0.106776  0.004614    ...
9             much  1.971340 -2.316020  0.147194 -0.641963 -0.280868    ...

            d14       d13       d12       d11       d10        d1         d0
0      0.432226 -0.023887 -0.246207  0.429862  0.268280  0.283950   0.218664   
1      0.702217 -0.516346  0.273179  0.662874  0.106199 -0.011592   0.057832   
2     -0.174151 -0.069734 -0.255887  0.070181 -0.163013  0.093490   0.028913
3     -0.189739 -0.089899 -0.048192  0.569139  0.595834  0.421905  -0.241777
4     -1.085993 -0.054178  1.156616 -1.449286  0.267787  0.677337   2.148856  
5     -0.004664 -0.414933 -0.346377 -0.214976  0.201621  0.063539  -0.331673
6      1.587940 -0.258819  1.396479  0.637493 -1.476619 -0.487518   0.864765    
7      0.190376  0.881103  0.966915  1.543105  1.974099 -0.807656   0.800163  
8     -0.181255  0.005893 -0.718905  0.373082  0.784821  0.393715  -0.000517  
9      1.348299  0.180225  1.686486  0.535154 -2.005099 -1.424234  -2.677770    
[9320 rows x 101 columns]
```
### Full Embedding Download
**`download(emb_name, set_name=None):`** Returns the full embedding, containing all uploaded word vectors in the shared embedding and saves the embedding as a .csv file in the current directory
  * **emb_name (str):** Title of the shared embedding
  * **set_name (str, opt):** Specify if multiple embeddings exist with the same emb_name

**For example:**
```python
>>> from vecshare import vecshare as vs
>>> vs.download('agriculture_40')
              text        d0        d1        d2        d3        d4  \
0              the  1.477964  0.016078 -0.193995  1.113142  0.765398   
1               of -0.048878 -0.597735  0.196982  0.220966  1.463818   
2               to  1.932197  1.587676 -0.321938 -0.592603  0.137684   
3               in  0.294486  1.061131 -0.119670  0.611166  0.436337   
4             said -0.609932 -0.481854  0.028189  0.755433 -0.493351   
5                a  0.750953  0.342545 -0.758257  0.381944  0.824879   
6              and  0.991821 -0.252496  0.011951  0.384948  0.505785   
7              mln  0.215208  3.330005  0.458480  0.484309  1.128098   
8               vs  0.512198  3.565070 -1.698517  0.813855 -0.002396   
9             dlrs -0.026384  1.905773  1.313683  0.825797  1.981671
```

## Advanced Setup
### Custom Signature Methods:
Additional signature methods can be included in the library by downloading the library and adding to the `signatures.py` file. To incorporate new signatures into future releases of the official VecShare library, fork and merge your changes with the github repository.

### Custom Indexers:
Custom indexers can be added by updating the `indexer.py` file.
```python
INDEXER      = <NEW INDEXER DATASET ID>
INDEX_FILE   = <NAME OF THE INDEX FILE>
EMB_TAG      = <EMB TAG>
```
