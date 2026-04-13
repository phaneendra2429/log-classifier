import re

def clean_log(text):
    text = str(text)

    # Remove timestamps (e.g., 2024-05-28T21:04:00Z)
    text = re.sub(r'\d{4}-\d{2}-\d{2}T.*?Z', ' ', text)

    # Remove log IDs (e.g., LOG-0001)
    text = re.sub(r'LOG-\d+', ' ', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text