# evaluate_testset.py
import re, ast, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.stem.snowball import SnowballStemmer
from collections import Counter, defaultdict

# ------------------ configuration ------------------
TOP_K          = 5
STEMMER_LANG   = "danish"
STOP_WORD_FILE = Path("data/danish_stopwords.txt")

TEST_CSV       = Path("dataset") / "test_dataset.csv"
PREP_FILE      = Path("preprocessor_lda.joblib")
CLS_FILE       = Path("classifiers_lda.joblib")
MLB_FILE       = Path("label_binarizer_lda.joblib")
QTY_CLS_FILE   = Path("quantity_classifiers_lda.joblib")
# --------------------------------------------------

stemmer = SnowballStemmer(STEMMER_LANG)

def load_stemmed_stopwords(path: Path) -> set[str]:
    sw = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            w = re.sub(r"[^a-z0-9åæø\s]", "", line.strip().lower())
            if w:
                sw.add(stemmer.stem(w))
    return sw

STOP_WORDS = load_stemmed_stopwords(STOP_WORD_FILE)

# ---------- helpers from train.py ----------
def preprocess_instruction(text: str) -> str:
    txt = re.sub(r"[^a-z0-9åæø\s]", " ", str(text).lower())
    return " ".join(stemmer.stem(tok) for tok in txt.split())

def parse_part_list(cell):
    try:
        lst = ast.literal_eval(cell)
        return lst if isinstance(lst, list) else [lst]
    except Exception:
        return []

def precision_at_k(y_true, proba, k=TOP_K):
    topk = np.argsort(proba, axis=1)[:, -k:]
    return np.mean([
        len(set(np.where(y_true[i] == 1)[0]) & set(topk[i])) / k
        for i in range(len(y_true))
    ])

def recall_at_k(y_true, proba, k=TOP_K):
    topk = np.argsort(proba, axis=1)[:, -k:]
    scores=[]
    for i in range(len(y_true)):
        t=set(np.where(y_true[i]==1)[0])
        if t:
            scores.append(len(t & set(topk[i]))/len(t))
    return np.mean(scores) if scores else 0.0

def f1_at_k(y_true, proba, k=TOP_K):
    p,r = precision_at_k(y_true,proba,k), recall_at_k(y_true,proba,k)
    return 2*p*r/(p+r) if (p+r) else 0.0

def weighted_proba_score(y_true, proba, k=TOP_K):
    topk = np.argsort(proba, axis=1)[:, -k:]
    vals=[]
    for i in range(len(y_true)):
        t=set(np.where(y_true[i]==1)[0])
        if t:
            vals.append(sum(proba[i,j] for j in topk[i] if j in t)/len(t))
    return np.mean(vals) if vals else 0.0

def iou_score(y_true, y_pred):
    return np.mean([
        1.0 if not (t or p) else len(t & p)/len(t | p)
        for t,p in (
            (set(np.where(y_true[i]==1)[0]), set(np.where(y_pred[i]==1)[0]))
            for i in range(len(y_true))
        )
    ])

def accuracy_counts(y_true_cnt,y_pred_cnt):
    mask = y_true_cnt > 0
    return np.mean((y_true_cnt[mask]==y_pred_cnt[mask]).astype(float))

def apply_quantity_safeguard(proba, qty_pred, k=TOP_K):
    topk = np.argsort(proba, axis=1)[:, -k:]
    for i,idxs in enumerate(topk):
        for j in idxs:
            if qty_pred[i,j] == 0:
                qty_pred[i,j] = 1
    return qty_pred
# ------------------------------------------------------------

def main():
    # ---------- 1) load test csv and preprocess (also from train.py) ----------
    df = pd.read_csv(TEST_CSV)
    df["Product ID (Product) (Product)"] = df["Product ID (Product) (Product)"].map(parse_part_list)
    df["Quantity"] = df["Quantity"].map(parse_part_list)
    df["Instructions"] = df["Instructions"].map(preprocess_instruction)

    # ---------- 2) laod trained models ----------
    preprocessor = joblib.load(PREP_FILE)
    classifiers  = joblib.load(CLS_FILE)
    mlb          = joblib.load(MLB_FILE)
    qty_clfs     = joblib.load(QTY_CLS_FILE)

    # ---------- 3) Features & targets ----------
    X = preprocessor.transform(df[["Instructions", "Primær Asset Produkt"]])

    Y_bin = mlb.transform(df["Product ID (Product) (Product)"])
    def cnt_vec(row):
        mapping = {p: q for p, q in zip(row["Product ID (Product) (Product)"], row["Quantity"])}
        return [mapping.get(cls, 0) for cls in mlb.classes_]
    Y_cnt = np.array([cnt_vec(r) for _, r in df.iterrows()]).astype(int)

    # ---------- 4) predictions ----------
    proba    = np.zeros(Y_bin.shape)
    qty_pred = np.zeros(Y_cnt.shape, dtype=int)

    for i, lbl in enumerate(mlb.classes_):
        if lbl in classifiers:
            proba[:, i] = classifiers[lbl].predict_proba(X)[:, 1]

    for i, lbl in enumerate(mlb.classes_):
        if lbl in qty_clfs:
            model = qty_clfs[lbl]
            qty_pred[:, i] = model.predict(X) if hasattr(model, "predict") else model

    qty_pred = apply_quantity_safeguard(proba, qty_pred)

    # ---------- 4b) include not seen before labels in metrics ----------
    unknown = sorted({p for parts in df["Product ID (Product) (Product)"] for p in parts}
                    - set(mlb.classes_))
    # total lable list
    all_labels = list(mlb.classes_) + unknown
    print(len(all_labels))


    if unknown:
        print(f"{len(unknown)} unkown labels → recall/qty_acc = 0, these are calculated within scores.")

        mlb_full = MultiLabelBinarizer(classes=all_labels).fit([])

        Y_bin = mlb_full.transform(df["Product ID (Product) (Product)"])

        def cnt_vec_full(row):
            mapping = {p: q for p, q in zip(row["Product ID (Product) (Product)"], row["Quantity"])}
            return [mapping.get(lbl, 0) for lbl in all_labels]

        Y_cnt_full = np.array([cnt_vec_full(r) for _, r in df.iterrows()]).astype(int)

        # -- proba / qty_pred uitbreiden med nullen --
        n = len(df)
        proba_full    = np.zeros((n, len(all_labels)))
        qty_pred_full = np.zeros_like(proba_full, dtype=int)

        old_idx = [all_labels.index(lbl) for lbl in mlb.classes_]
        proba_full[:, old_idx]    = proba
        qty_pred_full[:, old_idx] = qty_pred

        # vervang variabeler
        proba, qty_pred, Y_cnt = proba_full, qty_pred_full, Y_cnt_full
        # ----------------------------------------------------

    # ---------- 5) Metrics ----------
    n, C = proba.shape
    y_pred_bin = np.zeros_like(proba, dtype=int)
    topk = np.argsort(proba, axis=1)[:, -TOP_K:]
    y_pred_bin[np.repeat(np.arange(n), TOP_K), topk.ravel()] = 1

    metrics = {
        f"precision@{TOP_K}": precision_at_k(Y_bin, proba),
        f"recall@{TOP_K}"   : recall_at_k(Y_bin, proba),
        f"f1@{TOP_K}"       : f1_at_k(Y_bin, proba),
        "weighted"          : weighted_proba_score(Y_bin, proba),
        "iou"               : iou_score(Y_bin, y_pred_bin),
        "quantity_acc"      : accuracy_counts(Y_cnt, qty_pred)
    }

    print("\n===== Test-set metrics =====")
    for k, v in metrics.items():
        print(f"{k:>15}: {v:.3f}")


    true_sets = [set(np.where(Y_bin[i] == 1)[0]) for i in range(n)]
    pred_sets = [set(topk[i])                     for i in range(n)]

    tp   = Counter()
    gt   = Counter()
    pp   = Counter()
    conf = defaultdict(Counter)  # confusion[target][other]

    for i in range(n):
        g = true_sets[i]
        p = pred_sets[i]  # Rettede 'predsets' til 'pred_sets'
        for lbl in g:
            gt[lbl] += 1
            if lbl in p:
                tp[lbl] += 1
        for lbl in p:
            pp[lbl] += 1
        for wrong in p - g:
            for tgt in g:
                conf[tgt][wrong] += 1

    rows = []
    for lbl in range(C):
        if gt[lbl] == 0:
            continue
        rec  = tp[lbl] / gt[lbl]
        prec = tp[lbl] / pp[lbl] if pp[lbl] else 0.0
        mix  = ", ".join(f"{all_labels[o]} ({c})"  # Rettede 'mlb.classes' til 'all_labels'
                         for o, c in conf[lbl].most_common(3)) or "—"  # Rettede 'mostcommon' til 'most_common'
        rows.append((rec, lbl, prec, mix))

    rows.sort(reverse=True)  # højeste recall øverst

    print("\n===== Per-product performance (sorted by recall) =====")
    print(f"{'Product ID':<25} {'Recall':>7} {'Precision':>9}  Confused with")
    for rec, lbl, prec, mix in rows:
        print(f"{all_labels[lbl]:<25} {rec:7.3f} {prec:9.3f}  {mix}")


if __name__ == "__main__":
    main()
