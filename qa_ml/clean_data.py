import pandas as pd

INPUT_FILE = "data/raw/IncidentDetails.xlsx"
OUTPUT_FILE = "data/processed/tickets.csv"

def main():
    df = pd.read_excel(INPUT_FILE, sheet_name="Incident and QA Details")

    df["score"] = (df["score"] * 100).round(2)

    df["text"] = df["Summary"].astype(str) + " " + df["Description"].astype(str)

    cols_keep = [
        "Incident #", "Assigned Group", "Product Name", "Reported Source", "text", "score",
        "customer details", "correct summary", "Correct Template",
        "Template completed", "Complete and understandable troubleshooting notes",
        "Error message and or screenshot", "second ticket created", "Asset details",
        "Appropriate KBA article related", "Was FCR obtained", "Status update process",
        "Saved in progress first", "Sent to correct queue", "Correct resolution"
    ]
    df = df[cols_keep]

    df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()