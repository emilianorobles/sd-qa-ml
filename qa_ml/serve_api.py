import joblib
from fastapi import FastAPI
from qa_ml.schema import TicketIn, PredictionOut
from qa_ml.features import redact_pii, transform_text

bundle = joblib.load("models/identity_verif/model.joblib")
clf, tfw, tfc = bundle["clf"], bundle["tfw"], bundle["tfc"]

app = FastAPI(title="SD QA ML")

@app.post("/predict", response_model=PredictionOut)
def predict(item: TicketIn):
    txt = redact_pii(item.text)
    X = transform_text([txt], tfw, tfc)
    proba = clf.predict_proba(X)[0][1]
    pred = int(proba >= 0.5)
    feedback = []
    if not pred:
        feedback.append("Incluye una frase explícita de verificación de identidad.")
    return {"labels": {"identity_verification": {"pred": pred, "p": float(proba)}},
            "feedback_bullets": feedback}