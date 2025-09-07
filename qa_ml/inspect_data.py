from pathlib import Path
import pandas as pd

CSV = Path("data/processed/tickets.csv")

def main():
    df = pd.read_csv(CSV)
    print("Filas:", len(df))
    print("\nColumnas:", list(df.columns))

    print("\nScore (describe):")
    print(df["score"].describe())

    print("\nTop productos:")
    print(df["Product Name"].value_counts().head(5))
    print("\nTop canales:")
    print(df["Reported Source"].value_counts().head(5))

    qa_cols = [c for c in df.columns if c not in ["Incident #", "Assigned Group", "Product Name", "Reported Source", "text", "score"]]
    print("\nBalance por etiqueta QA (promedio ~ tasa de 1s):")
    print(df[qa_cols].mean().sort_values(ascending=False).round(3))

    print("\nNulos en text:", df["text"].isna().sum())

if __name__ == "__main__":
    main()