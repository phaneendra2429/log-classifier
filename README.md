# Log Classification AI Pipeline

## Overview
This project implements a lightweight AI pipeline to automatically classify system log entries into predefined root cause categories. It also generates structured summaries for each log and evaluates model performance using standard metrics.

The goal is to assist operations teams in quickly identifying and responding to system issues, reducing manual effort and improving response time.

---

## System Design

### High-Level Architecture

User Input (Logs)  
→ Frontend (HTML UI using Flask)  
→ Backend (Flask Application)  
→ Inference Layer  
   - Preprocessing  
   - TF-IDF Vectorization  
   - Model Prediction  
   - Confidence Handling  
   - Summary Generation  
→ Output (Prediction + Structured Summary)

---

### Training Pipeline

Dataset (Excel - log_dataset)  
→ Data Cleaning  
→ Text Preprocessing  
→ Feature Engineering (TF-IDF)  
→ Model Training (Logistic Regression)  
→ Evaluation (Accuracy, Precision, Recall, F1)  
→ Save Model (model.pkl)

---

### Inference Pipeline

New Log Input  
→ Preprocessing  
→ Vector Transformation  
→ Model Prediction  
→ Confidence Threshold Check  
→ Top-2 Predictions  
→ Structured Summary Generation  
→ Final Output

---

## Model Approach

I used a **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization approach combined with a **Logistic Regression classifier**.

### Why this approach?
- The dataset is relatively small (~120 samples), making deep learning models prone to overfitting.
- TF-IDF with character n-grams effectively captures patterns in logs such as:
  - error codes (401, 502, timeout)
  - keywords (token, database, API)
- Logistic Regression is fast, interpretable, and performs well on small datasets.

This combination provides a strong balance between performance, simplicity, and efficiency.

---

## Data Preprocessing

The dataset was provided as an Excel file with multiple sheets:
- `log_dataset` (actual training data)
- `root_cause_labels` (metadata)
- `README` (documentation)

### Steps performed:
1. Selected only the relevant sheet (`log_dataset`)
2. Dropped rows with missing:
   - `log_message`
   - `root_cause_label`
3. Removed duplicate records
4. Cleaned text:
   - Removed timestamps and IDs
   - Removed special characters
   - Converted text to lowercase
   - Normalized whitespace
5. Feature extraction using TF-IDF (character n-grams 3–5)

---

## Evaluation Results

The model was evaluated using a stratified train-test split (80/20).

### Metrics:

- Accuracy: 0.875  
- Precision (macro): ~0.90  
- Recall (macro): ~0.88  
- F1 Score (macro): ~0.88  

### Observations:
- Strong performance on well-defined categories (e.g., authentication errors)
- Lower performance on overlapping categories (e.g., network vs database issues)
- Model behaves realistically and avoids overfitting

---

## Structured Output

For each log entry, the system provides:

- Predicted root cause
- Confidence score
- Alternative prediction
- Structured summary including:
  - Issue type
  - Severity
  - Affected component
  - Suggested action

This transforms raw logs into actionable insights.

---

## Tradeoffs

### Simplicity vs Performance
- Used TF-IDF + Logistic Regression instead of deep learning
- Faster and easier to interpret but limited semantic understanding

### Small Dataset Constraints
- Limited training samples (~10–15 per class)
- Affects generalization for rare or unseen patterns

### Precision vs Safety
- Introduced confidence threshold
- Returns `UNKNOWN` instead of incorrect predictions when confidence is low

---

## Limitations

- Small dataset limits model generalization
- Struggles with unseen vendor-specific logs (e.g., Twilio, Stripe)
- No deep contextual understanding
- Summary generation is rule-based

---

## Productionization Plan

### Monitoring
- Track prediction confidence distribution
- Monitor prediction frequency per class
- Detect anomalies in log patterns

### Drift Detection
- Detect distribution shifts in incoming logs
- Trigger retraining when drift is detected

### Scaling
- Deploy as REST API using Flask or FastAPI
- Batch process logs for high-volume systems
- Use streaming tools like Kafka for real-time ingestion

### Reliability
- Implement fallback rule-based system
- Retry failed predictions
- Use caching (Redis) for repeated queries

### Future Improvements
- Increase dataset size
- Use transformer-based models (BERT)
- Enhance summary generation using LLMs

---

## How to Run

### 1. Setup Environment

python -m venv venv  
venv\Scripts\activate  
pip install -r requirements.txt  

---

### 2. Train Model

python src/train.py  

---

### 3. Run Application (Web UI)

python app.py  

Open in browser:  
http://127.0.0.1:5000/

---

### 4. Run CLI Version (Optional)

python main.py  

---

## Example

Input:
ERROR [payment-gateway] Upstream provider Twilio returned 502

Output:
Predicted Label: UNKNOWN  
Confidence: 0.45  
Alternative: RC-03  

Summary:
- Issue Type: RC-03  
- Severity: HIGH  
- Component: Service  
- Suggested Action: Verify third-party API or retry  

---

## Conclusion

This project demonstrates a practical and efficient approach to log classification using classical machine learning techniques. It combines prediction, confidence handling, and structured summaries with a simple UI, making it suitable for real-world operational use cases.