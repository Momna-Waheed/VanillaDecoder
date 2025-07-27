import pandas as pd
import re

# Config
CSV_PATH = "urdu-news-dataset-1M.csv"
TXT_OUTPUT = "cleaned_urdu_news.txt"

# Load dataset
df = pd.read_csv(CSV_PATH, encoding="utf-8")

# Combine Headline and News Text
df["Headline"] = df["Headline"].fillna("")
df["News Text"] = df["News Text"].fillna("")
df["text"] = df["Headline"] + " " + df["News Text"]

# Save raw count before processing
total_raw_lines = len(df)

# Function to clean text
def clean_text(text):
    text = re.sub(r'https?://\S+', '', text)             # Remove URLs
    text = re.sub(r'\s+', ' ', text)                     # Normalize whitespace
    text = re.sub(r'[�::::<>]', '', text)                # Remove junk symbols
    text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # Remove emoticons
    text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)  # Remove symbols & pictographs
    text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)  # Remove transport & map symbols
    text = re.sub(r'[\U0001F700-\U0001F77F]', '', text)  # Remove alchemical symbols
    text = re.sub(r'[\U0001F780-\U0001F7FF]', '', text)  # Geometric shapes extended
    text = re.sub(r'[\U0001F800-\U0001F8FF]', '', text)  # Supplemental arrows
    text = re.sub(r'[\U0001F900-\U0001F9FF]', '', text)  # Supplemental symbols & pictographs
    text = re.sub(r'[\U0001FA00-\U0001FA6F]', '', text)  # Chess symbols etc.
    text = re.sub(r'[^\x00-\x7F\u0600-\u06FF\s،۔ءآأؤإئًٌٍَُِّْٖٓٔٗ٘]', '', text)  # Keep Urdu + whitespace + basic punctuation
    return text.strip()

# Apply cleaning
df["text"] = df["text"].apply(clean_text)

# Filter Urdu-heavy lines
def is_urdu_heavy(text):
    urdu_chars = re.findall(r'[\u0600-\u06FF]', text)
    return len(urdu_chars) / len(text) > 0.5 if len(text) > 0 else False

df = df[df["text"].apply(is_urdu_heavy)]

# Remove duplicates
df = df.drop_duplicates(subset="text")

# Save cleaned lines to file
df["text"].to_csv(TXT_OUTPUT, index=False, header=False, encoding="utf-8")

# Print Stats
processed_lines = len(df)
print(f"📊 Total raw lines: {total_raw_lines}")
print(f"✅ Cleaned & saved lines: {processed_lines}")
print(f"📄 Output file: {TXT_OUTPUT}")
