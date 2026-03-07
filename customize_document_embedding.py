import os, sys
import warnings
warnings.filterwarnings(action="ignore")

from typing import List
import numpy as np
import pandas as pd

import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download("punkt_tab")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from snowflake_util import *

WORD_VEC_SIZE = 256
SG_WINDOW_SIZE = 10
CLUSTER_SIZE = 64

def extract_text(df):
    return df[["TITLE", "SELFTEXT", "TEXT"]], df["TOPIC"].values

def read_csv(file_path: str):
    df = pd.read_csv(file_path)
    return extract_text(df)

def custom_document_embedding(df):
    def make_set(documents: List) -> List:
        return [{word for word in sentence} for sentence in documents]
    
    def get_word2vec(documents: List):
        # Skip-Gram Model
        model = gensim.models.Word2Vec(
            documents,
            min_count=1,
            vector_size=WORD_VEC_SIZE,
            window=SG_WINDOW_SIZE,
            sg=1)
        
        X = []; y = []
        for word in model.wv.index_to_key:
            y.append(word)
            X.append(model.wv[word])
            
        return np.array(X), np.array(y)
    
    def cluster_labeling(X) -> List:
        kmeans = KMeans(n_clusters=CLUSTER_SIZE, random_state=4444)
        kmeans.fit(X)
        return kmeans.labels_
    
    def bag_of_words(docset: List, encoder: dict):
        embedded = []
        for words in docset:
            count = [0] * CLUSTER_SIZE
            for word in words:
                count[encoder[word]] += 1
                
            embedded.append(count)
    
        return np.array(embedded)
    
    documents = []
    for i in df.index:
        sentence = \
            (df.iloc[i, 0] + ' ') if df.iloc[i, 0] else ' ' +\
            (df.iloc[i, 1] + ' ') if df.iloc[i, 1] else ' ' +\
            df.iloc[i, 2] if df.iloc[i, 2] else ''
        sentence = sentence.lower().strip()
        documents.append(word_tokenize(sentence))
    
    docset = make_set(documents)
    X, y = get_word2vec(documents)
    word_label = cluster_labeling(X)
    encoder = {str(y[i]):int(word_label[i]) \
               for i in range(X.shape[0])}
    
    return bag_of_words(docset, encoder)

def plot_with_pca(docvec, subreddit):
    # binvec: (NUM_DATA, CLUSTER_SIZE)
    # subreddit: (NUM_DATA, )
    scaler = StandardScaler()
    docvec = scaler.fit_transform(docvec)
    pca = PCA(n_components=2)
    docvec_t = pca.fit_transform(docvec)
    
    labels = np.unique(subreddit)

    for label in labels:
        idx = np.array(subreddit) == label
        plt.scatter(docvec_t[idx,0],
                    docvec_t[idx,1],
                    label=label,
                    alpha=0.6)

    plt.legend(title="Subreddit")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

if __name__ == "__main__":
    #df = read_csv("data/sample.csv")
    conn = get_connection()
    df = get_posts(conn)
    df, subreddit = extract_text(df)
    
    docvec = custom_document_embedding(df)

    plot_with_pca(docvec, subreddit)
    
