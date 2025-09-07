import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV = Path("data/processed/tickets.csv")

def main():
    df = pd.read_csv(CSV)

    # Histograma simple
    plt.figure(figsize=(8,5))
    plt.hist(df["score"], bins=20, color="skyblue", edgecolor="black")
    plt.title("Distribución de Score QA")
    plt.xlabel("Score (%)")
    plt.ylabel("Número de tickets")
    plt.grid(axis="y", alpha=0.7)
    plt.show()

    # Boxplot para ver outliers
    plt.figure(figsize=(6,4))
    plt.boxplot(df["score"], vert=False, patch_artist=True)
    plt.title("Boxplot de Score QA")
    plt.xlabel("Score (%)")
    plt.show()

if __name__ == "__main__":
    main()