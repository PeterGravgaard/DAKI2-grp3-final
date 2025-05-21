from pathlib import Path
import re, pickle, pandas as pd
from collections import Counter
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
import hdbscan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ─── config ──────────────────────────────────────────────────
DATA_DIR   = Path('.')
TRAIN_CSV  = DATA_DIR / 'Data/trainingdatabentaxnew.csv'
FORTRO_CSV = DATA_DIR / 'Data/fortrolig_data_filtred_thermoplan.csv'
OUT_DIR    = Path('models_topk_sbert'); OUT_DIR.mkdir(exist_ok=True)

WO_COL  = 'Work Order'
TOP_K   = 5
PCA_DIM = 100  # ← increased PCA dimension
HDB_MIN_CL, HDB_MIN_SM = 2, 1  # ← relaxed clustering params
CONST_THRESH = 0.80
LR_KW = {'max_iter':1000, 'class_weight':'balanced'}
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

DIGIT_RE = re.compile(r'(\d+)')
def to_int64(s: pd.Series):
    return pd.to_numeric(s.astype(str).str.extract(DIGIT_RE)[0], errors='coerce').astype('Int64')

# ─── load data ───────────────────────────────────────────────
train = pd.read_csv(TRAIN_CSV)
fort  = pd.read_csv(FORTRO_CSV)
train['varenummer'] = to_int64(train['Supplier Item number (Product) (Product)'])
train['ProductID']  = to_int64(train['Product ID (Product) (Product)'])
train = train.dropna(subset=['Instructions','ProductID',WO_COL])
train['ProductID'] = train['ProductID'].astype(int)

# ─── build Fortroligt docs -----------------------------------
prio = ['Varenumer prioritet 1','Varenummer prioritet 2','Varenummer prioritet 3']
text_cols = ['Koder','KodeBeskrivelser','Fejl','Løsningsforslag','Fejl (service)']
agg = {}
for _,row in fort.iterrows():
    txt=' '.join(str(row[c]) for c in text_cols if pd.notnull(row.get(c))).strip()
    if not txt: continue
    for col in prio:
        v=row.get(col)
        if pd.notnull(v): agg.setdefault(int(v),[]).append(txt)
prod_docs = {v:' '.join(lst) for v,lst in agg.items()}

# ─── SBERT embeddings ----------------------------------------
print('Embedding product docs with SBERT…')
embedder = SentenceTransformer(MODEL_NAME)
X_docs = embedder.encode([prod_docs[v] for v in prod_docs], batch_size=64, show_progress_bar=True)

# ─── reduce for clustering ───────────────────────────────────
pca = PCA(n_components=PCA_DIM, random_state=42)
X_pca = pca.fit_transform(X_docs)

clu = hdbscan.HDBSCAN(min_cluster_size=HDB_MIN_CL, min_samples=HDB_MIN_SM, prediction_data=True)
labels = clu.fit_predict(X_pca)
var2cl = {v:int(l) for v,l in zip(prod_docs, labels)}
train['ClusterID'] = train['varenummer'].map(var2cl).fillna(-1).astype(int)

# ─── Save each cluster to CSV ────────────────────────────────
CLUSTER_OUT_DIR = OUT_DIR / 'clusters'
CLUSTER_OUT_DIR.mkdir(parents=True, exist_ok=True)
for cid, grp in train.groupby('ClusterID'):
    path = CLUSTER_OUT_DIR / f'cluster_{cid}.csv'
    grp.to_csv(path, index=False)
    print(f'Saved {len(grp)} rows to {path}')

# ─── Visualize clusters in 2D and 3D ─────────────────────────
print('Generating 2D and 3D cluster plots…')
pca_2d = PCA(n_components=2).fit_transform(X_docs)
plt.figure(figsize=(10, 7))
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=labels, cmap='tab10', s=30)
plt.title('Product Clusters (2D PCA)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.savefig(OUT_DIR / 'clusters_2d_pca.png')
plt.show()

pca_3d = PCA(n_components=3).fit_transform(X_docs)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2], c=labels, cmap='tab10', s=30)
ax.set_title('Product Clusters (3D PCA)')
ax.set_xlabel('PCA 1'); ax.set_ylabel('PCA 2'); ax.set_zlabel('PCA 3')
plt.savefig(OUT_DIR / 'clusters_3d_pca.png')
plt.show()

# ─── Embed Instructions ──────────────────────────────────────
print('Embedding Instructions…')
inst_vec = embedder.encode(train['Instructions'].tolist(), batch_size=128, show_progress_bar=True)
train['__vec_idx'] = range(len(train))

# ─── Train/Test Split ────────────────────────────────────────
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
tr_idx, va_idx = next(gss.split(train, groups=train[WO_COL]))
tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]

# ─── Cluster classifier ──────────────────────────────────────
clf_cluster = LogisticRegression(**LR_KW).fit(inst_vec[tr_df['__vec_idx']], tr_df['ClusterID'])
print('Cluster‑classifier acc:', accuracy_score(va_df['ClusterID'], clf_cluster.predict(inst_vec[va_df['__vec_idx']])))

# ─── Per‑cluster product models ──────────────────────────────
prod_models={}
for cid,grp in tr_df.groupby('ClusterID'):
    X=inst_vec[grp['__vec_idx']]
    y=grp['ProductID']
    freq=y.value_counts(normalize=True)
    top=freq.index[:TOP_K].tolist()
    if freq.iloc[0]>=CONST_THRESH or len(freq)<=TOP_K:
        prod_models[cid]=('freq',top)
        print(f'cluster {cid:3d}: freq fallback ({len(grp)} rows)')
        continue
    clf=LogisticRegression(**LR_KW).fit(X,y)
    prod_models[cid]=('model',clf,clf.classes_)
    print(f'cluster {cid:3d}: model ({len(freq)} classes, {len(grp)} rows)')

# ─── Prediction helper ───────────────────────────────────────
def topk(cid:int, emb):
    rec=prod_models.get(cid)
    if not rec: return []
    if rec[0]=='freq': return rec[1][:TOP_K]
    _,clf,classes=rec
    probs=clf.predict_proba(emb)[0]
    return [int(classes[i]) for i in probs.argsort()[::-1][:TOP_K]]

# ─── IOU Evaluation ──────────────────────────────────────────
print('Evaluating IOU@5…')
ious=[]
for wo,grp in va_df.groupby(WO_COL):
    true=set(grp['ProductID'])
    pred=set()
    for _,row in grp.iterrows():
        emb=inst_vec[row['__vec_idx']].reshape(1,-1)
        cid=int(clf_cluster.predict(emb)[0])
        pred.update(topk(cid,emb))
    if not pred: continue
    ious.append(len(true&pred)/len(true|pred))
print(f'Mean IOU@{TOP_K}: {sum(ious)/len(ious):.3f} over {len(ious)} WOs')

# ─── Save models ─────────────────────────────────────────────
for obj,name in [
    (embedder,'sbert_embedder.pkl'),
    (pca,'pca_reducer.pkl'),
    (clu,'hdbscan_clusterer.pkl'),
    (var2cl,'varnum_to_cluster.pkl'),
    (clf_cluster,'cluster_classifier.pkl'),
    (prod_models,'product_models_topk.pkl')]:
    with open(OUT_DIR/name,'wb') as f: pickle.dump(obj,f)
print('Saved models to',OUT_DIR)

# ─── Runtime inference ───────────────────────────────────────
def predict_topk(text:str,k:int=TOP_K):
    emb=embedder.encode([text])
    cid=int(clf_cluster.predict(emb)[0])
    return cid, topk(cid,emb)[:k]

if __name__ == '__main__':
    print(predict_topk('Maskinen giver flow fejl TX0115 og trækker ikke vand.'))
