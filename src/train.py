import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

from preprocess import clean_log

# ----------------------------
# Project paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "Flutterwave AI Engineer Assessment Dataset.xlsx"
MODEL_FILE = PROJECT_ROOT / "model" / "model.pkl"
METRICS_FILE = PROJECT_ROOT / "outputs" / "metrics.txt"

# ----------------------------
# Load Excel (ONLY correct sheet)
# ----------------------------
all_sheets = pd.read_excel(DATA_FILE, sheet_name=None)
print("Sheets found:", list(all_sheets.keys()))

# Use only actual dataset sheet
df = all_sheets["log_dataset"]

# ----------------------------
# Clean dataset
# ----------------------------
df = df.dropna(subset=["log_message", "root_cause_label"])
df = df.drop_duplicates()

print("Columns:", df.columns)
print("Total rows:", len(df))

# ----------------------------
# Column selection
# ----------------------------
TEXT_COL = "log_message"
LABEL_COL = "root_cause_label"

# ----------------------------
# Preprocess text
# ----------------------------
df[TEXT_COL] = df[TEXT_COL].apply(clean_log)

X = df[TEXT_COL]
y = df[LABEL_COL]

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# Model
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", report)

# ----------------------------
# Save model
# ----------------------------
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
joblib.dump((model, vectorizer), MODEL_FILE)

# ----------------------------
# Save metrics
# ----------------------------
METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(METRICS_FILE, "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(report)

print("\nTraining complete. Model saved.")
