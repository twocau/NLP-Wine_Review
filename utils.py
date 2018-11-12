import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from nltk.corpus import stopwords

#### Usage 1:
####    df = loadRawData() --> data = cleanAndTransform(df) --> saveDate(filename)
#### Usage 2:
####    data = loadCleanData()

FILE_NAMES = ['data\winemag-data_first150k.csv', 'data\winemag-data_130k-v2.csv']


def loadRawData(file_name=FILE_NAMES[0]):
    df = pd.read_csv(file_name)
    df = df.set_index('Unnamed: 0', drop=True)
    return df

def filterStopWords(words):
    return [w for w in words if not w in stopwords.words('english')]

# Use this only when you are dealing with raw data.
# 1. Remove rows with missing country information
# 2. Re-group country
# 3. Rescale points
# 4. Text cleaning (see ETL.ipynb)
## input: dataframe from loadRawData
## ouput: new dataframe
def cleanAndTransform(df):
    data = df.copy()
    to_drop = data.country.index[data.country.isnull()==True]
    data = data.drop(to_drop)
    to_drop = data.price.index[data.price.isnull()==True]
    data = data.drop(to_drop)
    data = data.reset_index() # Reset index
    data.country = data.country.apply(lambda c: c if c in ['US', 'Italy', 'France'] else 'Other') # Re-group
    data.points = data.points.apply(lambda x: x-80) # Rescale
    # Text cleaning:
    data['clean_des']= data.description # Copy description
    data.clean_des = data.clean_des.str.lower() # Use lowercase letters
    data.clean_des = data.clean_des.str.replace('[^\w\s]',' ') # Remove punctuations
    data.clean_des = data.clean_des.str.replace('\d+', '') # Remove numbers
    data.clean_des = data.clean_des.str.split() # Tokenization
    print('Removing stop words...')
    data.clean_des = data.clean_des.apply(lambda x: filterStopWords(x)) # Remove stop words
    return data

def saveData(data, filename='data/clean_v1.csv'):
    to_save = data.copy()
    to_save.clean_des = to_save.clean_des.str.join(' ')
    to_save.to_csv(filename, encoding='utf-8', index=False)
    
def loadCleanData(filename='data/clean_v1.csv', split=False):
    data = pd.read_csv(filename)
    if split:
        data.clean_des = data.clean_des.str.split()
    return data

def loadClusterData(df_file='data/cluster_df_v1.csv', km_file='data/cluster2_v1.pkl', 
                    w_file='data/cluster_words_v1.csv', wm_file='data/word_matrix_v1.csv'):
    df = pd.read_csv(df_file)
    km = joblib.load(km_file)
    words = pd.read_csv(w_file)['word'].values
    w_matrix = pd.read_csv(wm_file)
    return df, km, words, w_matrix


    