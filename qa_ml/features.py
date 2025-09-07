import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")
RE_PHONE = re.compile(r"\b\d{10}\b")
RE_INC   = re.compile(r"\bINC\d{6,}\b", re.I)

def redact_pii(text: str) -> str:
    text = RE_EMAIL.sub("<EMAIL>", text)
    text = RE_PHONE.sub("<PHONE>", text)
    text = RE_INC.sub("<INC>", text)
    return text.lower()

def build_vectorizers():
    tfw = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=80_000)
    tfc = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=3, max_features=100_000)
    return tfw, tfc

def fit_transform_text(texts, tfw, tfc):
    Xw = tfw.fit_transform(texts)
    Xc = tfc.fit_transform(texts)
    return hstack([Xw, Xc]), tfw, tfc

def transform_text(texts, tfw, tfc):
    return hstack([tfw.transform(texts), tfc.transform(texts)])