import pandas as pd
import numpy as np
from pathlib import Path

CSV_IN = Path("data/processed/tickets.csv")
CSV_OUT = Path("data/processed/tickets_augmented.csv")

def main():
    df = pd.read_csv(CSV_IN)

    # Semilla para reproducibilidad
    np.random.seed(42)

    new_scores = []
    for i in range(len(df)):
        # Distribución: 20% bajos, 40% medios, 40% altos
        r = np.random.rand()
        if r < 0.2:  # low
            score = np.random.randint(20, 50)
        elif r < 0.6:  # mid
            score = np.random.randint(51, 85)
        else:  # high
            score = np.random.randint(86, 100)
        new_scores.append(score)

        # Ajuste dinámico de QA labels
        qa_cols = [c for c in df.columns if c not in ["Incident #","Assigned Group","Product Name","Reported Source","text","score"]]
        if score < 50:
            flip_n = 3
        elif score < 85:
            flip_n = 2
        else:
            flip_n = 1

        flip_cols = np.random.choice(qa_cols, flip_n, replace=False)
        for col in flip_cols:
            df.at[i, col] = 0

    df["score"] = new_scores

    # Guardar nuevo dataset
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False, encoding="utf-8")
    print(f"Archivo guardado en {CSV_OUT.resolve()} con {len(df)} filas")

if __name__ == "__main__":
    main()