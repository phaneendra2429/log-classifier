import joblib
from src.preprocess import clean_log

# load model
model, vectorizer = joblib.load("model/model.pkl")


# ----------------------------
# Summary Generator
# ----------------------------
def generate_summary(log_text, label):
    log_lower = log_text.lower()

    # detect severity
    if "critical" in log_lower or "error" in log_lower:
        severity = "HIGH"
    elif "warn" in log_lower:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    # detect component
    if "api" in log_lower:
        component = "API"
    elif "database" in log_lower or "db" in log_lower:
        component = "Database"
    elif "auth" in log_lower or "token" in log_lower:
        component = "Authentication"
    else:
        component = "Service"

    # suggested action
    if label == "RC-01":
        action = "Check authentication tokens or permissions"
    elif label == "RC-02":
        action = "Check database connectivity or query performance"
    elif label == "RC-03":
        action = "Verify third-party API status or retry request"
    else:
        action = "Investigate logs and retry operation"

    return {
        "issue_type": label,
        "severity": severity,
        "component": component,
        "suggested_action": action
    }


# ----------------------------
# Prediction Function
# ----------------------------
def predict_log(log_text):
    cleaned = clean_log(log_text)
    vec = vectorizer.transform([cleaned])

    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    top_idx = probs.argsort()[-2:][::-1]

    top_label = classes[top_idx[0]]
    top_prob = probs[top_idx[0]]

    # handle edge case
    if len(probs) > 1:
        second_label = classes[top_idx[1]]
        second_prob = probs[top_idx[1]]
    else:
        second_label = "N/A"
        second_prob = 0

    # confidence threshold
    if top_prob < 0.4:
        final_label = "UNKNOWN"
    else:
        final_label = top_label

    # generate summary
    summary = generate_summary(log_text, top_label)

    return {
        "log": log_text,
        "predicted_label": final_label,
        "confidence": round(top_prob, 2),
        "alternative": second_label,
        "alternative_confidence": round(second_prob, 2),
        "summary": summary
    }


# ----------------------------
# Test Run
# ----------------------------
if __name__ == "__main__":
    test_log = "ERROR database connection timeout after 30 seconds"
    result = predict_log(test_log)
    print(result)