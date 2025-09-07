# qa_ml/train_identity.py
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.sparse import hstack

CSV = Path("data/processed/tickets.csv")
MODEL_DIR = Path("models/correct_resolution")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABEL = "Correct resolution"

def main():
    df = pd.read_csv(CSV).dropna(subset=["text", LABEL])
    y = df[LABEL].astype(int).values
    texts = df["text"].astype(str).values

    # Vectorización palabra y carácter (robusto a typos)
    tf_w = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=80_000)
    tf_c = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=3, max_features=100_000)

    Xw = tf_w.fit_transform(texts)
    Xc = tf_c.fit_transform(texts)
    X = hstack([Xw, Xc])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    base = LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced", n_jobs=-1)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Xtr, ytr)

    preds = clf.predict(Xte)
    print(classification_report(yte, preds, digits=3))

    joblib.dump({"clf": clf, "tf_w": tf_w, "tf_c": tf_c}, MODEL_DIR / "model.joblib")
    print("Modelo guardado en", (MODEL_DIR / "model.joblib").resolve())

if __name__ == "__main__":
    main()