from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

def make_binary_clf():
    base = LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced", n_jobs=-1)
    return CalibratedClassifierCV(base, method="isotonic", cv=3)