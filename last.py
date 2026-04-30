import pdfplumber
import os
import re
import nltk
import json

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from collections import Counter
import matplotlib.pyplot as plt

# NEW IMPORTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# डाउनलोड (only first time)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# 📁 Folder path
folder_path = r"D:\NLP\reports"
cache_file = "year_texts.json"

# -------------------------------
# STEP 0: Helper functions
# -------------------------------
def is_valid_word(word):
    if re.search(r'(.)\1{1,}', word):
        return False
    if len(word) <= 2 or len(word) > 15:
        return False
    return True

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def tokenize_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words and is_valid_word(w)]

def lemmatize_words(words):
    return [lemmatizer.lemmatize(w) for w in words]

# -------------------------------
# STEP 1: Extract or Load Cache
# -------------------------------
if os.path.exists(cache_file):
    print("Loading from cache...")
    with open(cache_file, "r", encoding="utf-8") as f:
        year_texts = json.load(f)
else:
    print("Extracting PDFs...")
    year_texts = {}

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            year = file.replace(".pdf", "")
            file_path = os.path.join(folder_path, file)

            text = ""
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            text += t + " "
                year_texts[year] = text
            except Exception as e:
                print(f"Error reading {file}: {e}")

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(year_texts, f)

print("Extraction Done!")

# -------------------------------
# STEP 2: Preprocessing
# -------------------------------
processed_texts = {}

for year, text in year_texts.items():
    text = clean_text(text)
    tokens = tokenize_text(text)
    lemmas = lemmatize_words(tokens)
    processed_texts[year] = " ".join(lemmas)

years = sorted(processed_texts.keys(), key=int)
documents = [processed_texts[y] for y in years]

# -------------------------------
# STEP 3: TF-IDF (Important Words)
# -------------------------------
print("\n🔹 TF-IDF Important Words:\n")

tfidf = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf.fit_transform(documents)
feature_names = tfidf.get_feature_names_out()

for i, year in enumerate(years):
    scores = tfidf_matrix[i].toarray()[0]
    top_indices = scores.argsort()[-10:][::-1]

    print(f"\nYear {year}:")
    for idx in top_indices:
        print(f"{feature_names[idx]} ({scores[idx]:.3f})")

# -------------------------------
# STEP 4: LDA Topic Modeling
# -------------------------------
print("\n🔹 LDA Topics:\n")

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

def print_topics(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic {topic_idx + 1}:")
        top_words = [feature_names[i] for i in topic.argsort()[-n_top_words:]]
        print(", ".join(top_words))

print_topics(lda, feature_names)

# -------------------------------
# STEP 5: Topic Distribution per Year
# -------------------------------
topic_distribution = lda.transform(tfidf_matrix)

print("\n🔹 Topic Distribution per Year:\n")
for i, year in enumerate(years):
    print(f"{year}: {topic_distribution[i]}")

# -------------------------------
# STEP 6: (Optional) Seed Scoring (Your Old Logic)
# -------------------------------
seeds = {
    "women_safety": ["violence", "harassment", "crime", "helpline"],
    "nutrition": ["nutrition", "malnutrition", "food", "anganwadi"],
}

seed_scores = {}

for year in years:
    words = processed_texts[year].split()
    freq = Counter(words)

    seed_scores[year] = {}
    for seed, keywords in seeds.items():
        score = sum(freq[k] for k in keywords)
        seed_scores[year][seed] = score

# -------------------------------
# STEP 7: Visualization
# -------------------------------
for seed in seeds:
    values = [seed_scores[y][seed] for y in years]
    plt.plot(years, values, label=seed)

plt.title("Seed Trends Over Time")
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
