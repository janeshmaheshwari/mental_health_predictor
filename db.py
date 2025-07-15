from pathlib import Path
import sqlite3
import pandas as pd



conn = sqlite3.connect("mental_health.db")
df = pd.read_sql("SELECT * FROM responses", conn)

labels = ["Depression", "Anxiety", "Insomnia", "Social Anxiety", "Psychotic Symptoms"]
print("Label Variety Check:\n")
for label in labels:
    if label in df.columns:
        print(f"{label}: {df[label].unique()}")
    else:
        print(f"{label}: ❌ Column not found")

conn.close()


DB_PATH = Path(__file__).with_name("mental_health.db")
CSV_PATH = Path(__file__).with_name("responses.csv")

COL_ORDER = [
    "timestamp", "source",
    "Age", "Gender", "Educational Level", "Employment Status",
    "How many hours do you sleep per day (on average)?",
    "How many hours do you use mobile, computer, or TV daily?",
    "If you work or study, how many hours do you do so daily?",
    "Do you engage in any physical activity (e.g., walking, gym, yoga) daily?",
    "Do you usually use screens after 10 PM at night?",
    "How would you rate your caffeine intake?",
    "How many meals do you eat per day (on average)?",
    "Self Esteem",
    "Depression", "Anxiety", "Insomnia", "Social Anxiety", "Psychotic Symptoms"
]

def _ensure_table():
    quoted = [f'"{col}" TEXT' for col in COL_ORDER]
    with sqlite3.connect(DB_PATH) as con:
        con.execute(f"CREATE TABLE IF NOT EXISTS responses ({', '.join(quoted)});")

def append_row(record: dict):
    _ensure_table()
    ordered = {col: record.get(col, "") for col in COL_ORDER}
    df_row = pd.DataFrame([ordered])
    write_header = not CSV_PATH.exists()
    df_row.to_csv(CSV_PATH, mode="a", header=write_header, index=False)
    with sqlite3.connect(DB_PATH) as con:
        df_row.to_sql("responses", con, if_exists="append", index=False)

def load_data():
    _ensure_table()
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql("SELECT * FROM responses", con)