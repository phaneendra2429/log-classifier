import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

from preprocess import clean_log

# ----------------------------
# Load Excel (ONLY correct sheet)
# ----------------------------
file_path = "data/Flutterwave AI Engineer Assessment Dataset.xlsx"

all_sheets = pd.read_excel(file_path, sheet_name=None)
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
joblib.dump((model, vectorizer), "model/model.pkl")

# ----------------------------
# Save metrics
# ----------------------------
with open("outputs/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(report)

print("\nTraining complete. Model saved.")