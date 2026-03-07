import os, sys
import warnings
warnings.filterwarnings(action="ignore")

from typing import List
import numpy as np
import pandas as pd

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download("punkt_tab")
nltk.download("stopwords")

from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import silhouette_score, davies_bouldin_score

import matplotlib.pyplot as plt

from snowflake_util import *

WORD_VEC_SIZE = [64, 128, 256]
CLUSTER_SIZE = 8
N_CLUSTERS = 8
SG_WINDOW_SIZE = 5
WORD_MIN_COUNT = 1

STOP_WORDS = set(stopwords.words("english"))

def extract_text(df):
    return df[["TITLE", "SELFTEXT", "TEXT"]], df["TOPIC"].values

def read_csv(file_path: str):
    df = pd.read_csv(file_path)
    return extract_text(df)

def cluster_vectors(vecs: np.ndarray):
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    return km.fit_predict(vecs), km

def evaluate(vecs: np.ndarray, labels: np.ndarray) -> dict:
    sil = silhouette_score(vecs, labels, metric="cosine")
    db  = davies_bouldin_score(vecs, labels)

    cohesions = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            cohesions.append(1.0)
            continue
        sub = vecs[idx]
        sim = np.dot(sub, sub.T)
        mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
        cohesions.append(sim[mask].mean())

    return {
        "silhouette":       round(float(sil), 4),
        "davies_bouldin":   round(float(db), 4),
        "avg_intra_cosine": round(float(np.mean(cohesions)), 4),
        "n_clusters":       int(np.unique(labels).size),
    }

def custom_document_embedding(df, i):
    def make_set(documents: List) -> List:
        return [{word for word in sentence} for sentence in documents]
    
    def get_word2vec(documents: List):
        with open(os.devnull, 'w') as devnull:
            old_stderr = sys.stderr
            sys.stderr  = devnull
            try:
                # Skip-Gram Model
                model = gensim.models.Word2Vec(
                    documents,
                    min_count=WORD_MIN_COUNT,
                    vector_size=WORD_VEC_SIZE[i],
                    window=SG_WINDOW_SIZE,
                    sg=1,
                    workers=1,
                )
            finally:
                sys.stderr = old_stderr
        
        X = []; y = []
        for word in model.wv.index_to_key:
            y.append(word)
            X.append(model.wv[word])
            
        return np.array(X), np.array(y)
    
    def cluster_labeling(X) -> List:
        kmeans = KMeans(n_clusters=CLUSTER_SIZE, random_state=4444, n_init=10)
        kmeans.fit(X)
        return kmeans.labels_
    
    def bag_of_words(documents: List, encoder: dict):
        embedded = []
        for words in documents:
            count = [0] * CLUSTER_SIZE
            for word in words:
                if word in encoder:
                    count[encoder[word]] += 1
            embedded.append(count)
        return np.array(embedded)
    
    documents = []
    for _, row in df.iterrows():
        sentence = " ".join([
            str(row.iloc[0]) if pd.notna(row.iloc[0]) else "",
            str(row.iloc[1]) if pd.notna(row.iloc[1]) else "",
            str(row.iloc[2]) if pd.notna(row.iloc[2]) else "",
        ])
        
        tokens = [
            t for t in simple_preprocess(sentence, deacc=True)
            if t not in STOP_WORDS and len(t) > 2
        ]
        documents.append(tokens)
    
    # docset = make_set(documents)
    
    X, y = get_word2vec(documents)
    word_label = cluster_labeling(X)
    encoder = {str(y[i]):int(word_label[i]) \
               for i in range(X.shape[0])}

    bow = bag_of_words(documents, encoder)

    tfidf = TfidfTransformer()
    bow_tfidf = tfidf.fit_transform(bow).toarray()
    bow_norm = normalize(bow_tfidf,  norm="l2")
    return bow_norm

def plot_with_pca(docvec, subreddit, i, ax):
    # binvec: (NUM_DATA, CLUSTER_SIZE)
    # subreddit: (NUM_DATA, )
    zscore = np.abs(stats.zscore(docvec))
    mask = (zscore < 3).all(axis=1)
    docvec = docvec[mask]
    subreddit = subreddit[mask]

    # scaler = StandardScaler()
    scaler = RobustScaler()
    docvec = scaler.fit_transform(docvec)
    pca = PCA(n_components=2)
    docvec_t = pca.fit_transform(docvec)
    
    labels = np.unique(subreddit)

    for label in labels:
        idx = np.array(subreddit) == label
        ax.scatter(docvec_t[idx,0],
                    docvec_t[idx,1],
                    label=label,
                    s=8,
                    alpha=0.6)

    ax.set_title(f"Dimension = {WORD_VEC_SIZE[i]}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="upper right", title="True Topic")

if __name__ == "__main__":
    #df = read_csv("data/sample.csv")
    conn = get_connection()
    df = get_posts(conn)
    df, subreddit = extract_text(df)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Word2Vec & BoW - PCA 2D Projection", fontweight="bold")

    for i, ax in enumerate(axes):
        docvec = custom_document_embedding(df, i)
        plabel, _ = cluster_vectors(docvec)
        metrics = evaluate(docvec, plabel)
        print(f"Metrics: {metrics}")
        
        plot_with_pca(docvec, subreddit, i, ax)

    plt.show()
    
