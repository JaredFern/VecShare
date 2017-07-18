from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium import webdriver
from bs4 import BeautifulSoup
import datadotworld as dw

INDEXER_URL  = 'https://data.world/jaredfern/vecshare-indexer'  # URL for index & signature file 
DATASETS_URL = 'https://data.world/datasets/vecshare'           # URL for uploaded embeddings
BASE_URL     = 'https://data.world'                             

wd = webdriver.Chrome()
wd.get(DATASETS_URL)
WebDriverWait(wd,5).until(
    EC.visibility_of_element_located((By.CLASS_NAME, 'dw-dataset-name DatasetCard__name___2U4-H')))

source  = wd.page_source
soup    = BeautifulSoup(source)
sets    = soup.find_all('a', 'dw-dataset-name DatasetCard__name___2U4-H')
urls    = [dw_set["href"] for dw_set in sets]   

index_file = dw.load_query('vecshare-indexer')

 