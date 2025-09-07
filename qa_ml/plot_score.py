import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/tickets_augmented.csv")

plt.figure(figsize=(8,5))
plt.hist(df["score"], bins=20, color="lightgreen", edgecolor="black")
plt.title("Distribución del Score (augmented)")
plt.xlabel("Score (%)")
plt.ylabel("Número de tickets")
plt.grid(axis="y", alpha=0.7)
plt.show()