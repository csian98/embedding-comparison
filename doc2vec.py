import numpy as np
import pandas as pd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from snowflake_util import get_connection, get_posts

'''
Parameters:
vector_size: dimensionality
min_count: ignore words with total frequency < min_count
dm: 1 -> PV-DM (predict word from context + doc), 0 -> PV-DBOW(predict context from doc)
window: how many words to consider(PV-DM only)
'''
CONFIGS = {
    "Config A (dim=64)": dict(
        vector_size=64,
        min_count=2,
        epochs=20,
        dm=1,        # PV-DM  — uses word-order context
        window=5,
        workers=4,
        seed=42,
    ),
    "Config B (dim=128)": dict(
        vector_size=128,
        min_count=3,
        epochs=40,
        dm=0,        # PV-DBOW — faster; often stronger on short social-media posts
        workers=4,
        seed=42,
    ),
    "Config C (dim=256)": dict(
        vector_size=256,
        min_count=1,  # keep rare / niche vocabulary
        epochs=60,
        dm=1,
        window=8,
        workers=4,
        seed=42,
    ),
}

def build_tagged_docs(df: pd.DataFrame) -> list[TaggedDocument]:
    """Convert each row into a gensim TaggedDocument keyed by its integer index."""
    tagged = []
    for idx, row in df.iterrows():
        text = str(row["SELFTEXT"]) if pd.notna(row["SELFTEXT"]) else ""
        tokens = simple_preprocess(text, deacc=True)   # lowercase, strips punctuation and numbers
        tagged.append(TaggedDocument(words=tokens, tags=[str(idx)]))
    return tagged

def train_doc2vec(tagged_docs: list[TaggedDocument], config: dict) -> Doc2Vec:
    model = Doc2Vec(**config)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def extract_vectors(model: Doc2Vec, tagged_docs: list[TaggedDocument]) -> np.ndarray:
    """Return L2-normalised document vectors.
    After L2 normalisation, Euclidean distance == cosine distance,
    so standard K-Means minimises the cosine objective."""
    vecs = np.array([model.dv[doc.tags[0]] for doc in tagged_docs])
    return normalize(vecs, norm="l2")

N_CLUSTERS = 6
def cluster_vectors(vecs: np.ndarray):
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    return km.fit_predict(vecs), km


def evaluate(vecs: np.ndarray, labels: np.ndarray) -> dict:
    sil = silhouette_score(vecs, labels, metric="cosine") # how similar a data point is to its own cluster  compared to other clusters (higher -> better)
    db  = davies_bouldin_score(vecs, labels) # the average similarity measure of each cluster with its most similar cluster (lower -> better)

    cohesions = [] # average cosine similarity between all pairs of documents within the same cluster(higher -> better)
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


def plot_clusters(vecs: np.ndarray, pred_labels: np.ndarray, true_labels: np.ndarray,
                  title: str, ax: plt.Axes, sample: int = 2000):

    # subsample for large datasets
    if len(vecs) > sample:
        idx = np.random.choice(len(vecs), sample, replace=False)
        vecs, pred_labels, true_labels = vecs[idx], pred_labels[idx], true_labels[idx]

    # PCA to 2D and assign colors for visualization
    proj = PCA(n_components=2, random_state=42).fit_transform(vecs)
    unique_topics = sorted(set(true_labels))
    topic_colors = {t: cm.tab10(i / max(len(unique_topics) - 1, 1))
                    for i, t in enumerate(unique_topics)}

    
    for topic in unique_topics:
        mask = true_labels == topic # boolean array 
        ax.scatter(proj[mask, 0], proj[mask, 1], s=8, alpha=0.6,
                   color=topic_colors[topic], label=topic)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize=6, markerscale=2, loc="upper right", title="True Topic")

STOPWORDS = {
    "the","a","an","is","it","in","on","of","to","and","or","for","with",
    "this","that","be","at","are","was","were","as","by","from","but","not",
    "have","has","had","been","will","its","their","they","we","you","i",
    "he","she","my","your","im","ive","dont","cant","wont","didnt",
    "her","his","him","out","what","like","just","one","all","how",
    "know","about","would","more","could","said","into","also","even",
    "when","then","there","which","who","get","got","so","if","up",
    "do","did","use","can","now","than","over","after","because","see",
    "some","any","other","them","our","through","before","only","still",
    "where","really","something","should","around","back","again","way",
    "me","no","off","never","am","myself","these"

}

def top_terms_per_cluster(df: pd.DataFrame, labels: np.ndarray, n_terms: int = 15) -> dict:
    result = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        tokens = []
        for i in idx:
            tokens.extend(simple_preprocess(str(df.iloc[i]["SELFTEXT"])))
        filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

        if not filtered:
            result[c] = {"_empty_cluster": len(idx)}  # report size even if no terms
        else:
            result[c] = pd.Series(filtered).value_counts().head(n_terms).to_dict()
    return result

def main():
    
    con = get_connection()
    posts_df = get_posts(con)
    con.close()

    # drop nulls 
    posts_df = posts_df[posts_df["SELFTEXT"].notna()].reset_index(drop=True)
    

    # Build tagged documents
    print("\nBuilding tagged documents...")
    tagged_docs = build_tagged_docs(posts_df)

    # ── Train → cluster → evaluate ─────────────────────────────────────────
    all_vectors = {}
    all_labels  = {}
    all_results = {}

    for name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print(f"  Params: vector_size={cfg['vector_size']}, "
              f"min_count={cfg['min_count']}, epochs={cfg['epochs']}, "
              f"dm={cfg['dm']}")

        model  = train_doc2vec(tagged_docs, cfg)
        vecs   = extract_vectors(model, tagged_docs)

        print(f"  Clustering into {N_CLUSTERS} clusters...")
        labels, _ = cluster_vectors(vecs)
        metrics   = evaluate(vecs, labels)
        print(f"  Metrics: {metrics}")

        all_vectors[name] = vecs
        all_labels[name]  = labels
        all_results[name] = metrics

    # Summary table 
    metrics_df = pd.DataFrame(all_results).T
    print("\n\n" + "="*70)
    print("COMPARATIVE EVALUATION SUMMARY")
    print("="*70)
    print(metrics_df.to_string())

    # Cluster inspection
    print("\n\n" + "="*70)
    print("CLUSTER CONTENT INSPECTION")
    print("="*70)

    for name in CONFIGS:
        print(f"\n--- {name} ---")
        terms = top_terms_per_cluster(posts_df, all_labels[name])

        # print cluster size distribution
        for c in sorted(terms.keys()):
            idx = np.where(all_labels[name] == c)[0]
            top = ", ".join(list(terms[c].keys())[:8])
            print(f"  Cluster {c} ({len(idx):>4} docs)  top terms: {top}")

    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Doc2Vec Clustering — PCA 2-D Projection (cosine distance)",
        fontsize=13, fontweight="bold"
    )
    true_labels = posts_df["TOPIC"].values if "TOPIC" in posts_df.columns else np.zeros(len(posts_df))

    for ax, (name, vecs) in zip(axes, all_vectors.items()):
        short = name.split("—")[0].strip()
        plot_clusters(vecs, all_labels[name], true_labels, short, ax)

    plt.tight_layout()
    out_path = "./images/doc2vec_clusters.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nCluster plot saved {out_path}")

    # Best-config recommendation
    # composite = silhouette + avg_intra_cosine − 0.3 × davies_bouldin
    composite = {
        n: (
            metrics_df.loc[n, "silhouette"]
            + metrics_df.loc[n, "avg_intra_cosine"]
            - 0.3 * metrics_df.loc[n, "davies_bouldin"]
        )
        for n in metrics_df.index
    }
    best = max(composite, key=composite.get)

    print("\n\n" + "="*70)
    print("  RECOMMENDATION")
    print("="*70)
    print(f"\n  Best configuration : {best}")
    print(f"  Composite scores   : {composite}")


if __name__ == "__main__":
    main()