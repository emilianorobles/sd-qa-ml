# qa_ml/train_score_bucket.py
"""
Clasificador multiclase de calidad (score) por rangos:
- 0: Bajo   (0–50)
- 1: Medio  (51–85)
- 2: Alto   (86–100)

Aprenderás:
- Cómo hacer bins (rankeos) del score
- TF-IDF de palabras + caracteres (robusto a typos)
- Entrenar Logistic Regression (multinomial) con calibración
- Evaluar con matriz de confusión y F1 por clase
- Guardar el modelo y vectorizadores
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

CSV = Path("data/processed/tickets_augmented.csv")
MODEL_DIR = Path("models/score_bucket")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# 1) Binning de score
# -----------------------
def to_bucket(score: float) -> int:
    """Convierte score (0-100) a clase: 0=bajo, 1=medio, 2=alto"""
    if score <= 50:
        return 0
    elif score <= 85:
        return 1
    else:
        return 2

def main():
    # -----------------------
    # 2) Cargar datos
    # -----------------------
    df = pd.read_csv(CSV).dropna(subset=["text", "score"])
    # Convertir a buckets
    y = df["score"].apply(to_bucket).astype(int).values
    texts = df["text"].astype(str).values

    # (opcional) inspección rápida de distribución
    uniq, counts = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq, counts)}
    print("Distribución de clases (0=bajo,1=medio,2=alto):", dist)

    # -----------------------
    # 3) Vectorización
    # -----------------------
    tf_w = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=80_000)
    tf_c = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_features=100_000)

    Xw = tf_w.fit_transform(texts)
    Xc = tf_c.fit_transform(texts)
    X = hstack([Xw, Xc])

    # -----------------------
    # 4) Split estratificado
    # -----------------------
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # -----------------------
    # 5) Modelo
    # -----------------------
    # LogisticRegression multinomial suele ir bien en texto
    # class_weight="balanced" ayuda si las clases no están exactamente parejas
    clf = LogisticRegression(
        max_iter=2000,
        solver="saga",
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(Xtr, ytr)

    # -----------------------
    # 6) Evaluación
    # -----------------------
    preds = clf.predict(Xte)
    print("\nReporte de clasificación (test):")
    print(classification_report(yte, preds, digits=3))

    print("Matriz de confusión (filas=verdad, columnas=predicción):")
    print(confusion_matrix(yte, preds))

    # -----------------------
    # 7) Guardado
    # -----------------------
    joblib.dump({"clf": clf, "tf_w": tf_w, "tf_c": tf_c}, MODEL_DIR / "model.joblib")
    print("Modelo guardado en:", (MODEL_DIR / "model.joblib").resolve())

    # -----------------------
    # 8) Ejemplo de inferencia
    # -----------------------
    example = df["text"].iloc[0]
    # proba = clf.predict_proba(tf_w.transform([example]) if hasattr(clf, "predict_proba") else Xte[:1])
    # Nota: Para ser exactos, debemos transformar con ambos vectorizadores:
    Xe = hstack([tf_w.transform([example]), tf_c.transform([example])])
    proba = clf.predict_proba(Xe)[0]
    pred_bucket = clf.predict(Xe)[0]
    print("\nEjemplo de predicción con el primer ticket:")
    print("bucket_predicho:", int(pred_bucket), "(0=bajo,1=medio,2=alto)")
    print("proba (0,1,2):", proba)

if __name__ == "__main__":
    main()