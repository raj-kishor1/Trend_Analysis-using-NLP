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

# डाउनलोड (only first time)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
# 📁 Folder path
folder_path = r"D:\NLP\reports"
cache_file = "year_texts.json"
year_texts = {}

def is_valid_word(word):
    # Remove words with repeating pattern like 'ooff', 'tthhee'
    if re.search(r'(.)\1{1,}', word):  
        return False
    
    # Remove very short/meaningless
    if len(word) <= 2:
        return False
    
    # Remove weird long words
    if len(word) > 15:
        return False
    
    return True

# 🔹 STEP 1: Extract text from PDFs
if os.path.exists(cache_file):
    print("Loading from cache...")

    with open(cache_file, "r", encoding="utf-8") as f:
        year_texts = json.load(f)

else:
    print("Extracting from PDFs...")

    year_texts = {}

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            year = file.replace(".pdf", "")
            file_path = os.path.join(folder_path, file)

            print(f"Reading {file}...")

            text = ""

            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + " "

                year_texts[year] = text

            except Exception as e:
                print(f"Error reading {file}: {e}")

    # 💾 Save cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(year_texts, f)

print("Done!")

# 🔹 STEP 2: Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

cleaned_year_texts = {}

for year, text in year_texts.items():
    cleaned_year_texts[year] = clean_text(text)

print("\nCLEANED SAMPLE:\n")
print(cleaned_year_texts["2010"][:500])


# 🔹 STEP 3: Tokenization + Stopword removal
stop_words = set(stopwords.words('english'))

def tokenize_text(text):
    words = word_tokenize(text)
    filtered_words = [
    w for w in words
    if w not in stop_words
    and is_valid_word(w)
    ]
    return filtered_words

tokenized_year_texts = {}

for year, text in cleaned_year_texts.items():
    tokenized_year_texts[year] = tokenize_text(text)

print("\nTOKENIZED SAMPLE:\n")
print(tokenized_year_texts["2010"][:50])

#-----------------------------------------------
def lemmatize_words(words):
    return [lemmatizer.lemmatize(w) for w in words]
lemmatized_year_texts = {}

for year, words in tokenized_year_texts.items():
    lemmatized_year_texts[year] = lemmatize_words(words)

print("\nLEMMATIZED SAMPLE:\n")
print(lemmatized_year_texts["2010"][:50])

#----------------------------------------------------   setp5 word freq
word_freq_by_year = {}

for year, words in lemmatized_year_texts.items():
    word_freq_by_year[year] = Counter(words)

print("\nWORD FREQUENCY SAMPLE:\n")
print(word_freq_by_year["2010"].most_common(20))

#---------------------------------------------------   step 6 seeding

seeds = {
    "women_safety": {
        "violence": 1.0,
        "harassment": 1.0,
        "crime": 0.9,
        "helpline": 0.8,
        "protection": 0.7
    },
    "child_development": {
        "child": 1.0,
        "education": 0.9,
        "growth": 0.7,
        "development": 0.6
    },
    "nutrition": {
        "nutrition": 1.0,
        "malnutrition": 1.0,
        "icds": 0.9,
        "anganwadi": 0.8,
        "food": 0.6
    },
    "women_empowerment": {
        "empowerment": 1.0,
        "employment": 0.8,
        "training": 0.7,
        "self": 0.5
    },
    "legal_support": {
        "act": 1.0,
        "law": 1.0,
        "policy": 0.8,
        "protection": 0.7
    }
}

seed_scores_by_year = {}

for year, freq_dict in word_freq_by_year.items():
    seed_scores = {}

    for seed, word_weights in seeds.items():
        score = 0

        for word, weight in word_weights.items():
            score += freq_dict.get(word, 0) * weight

        seed_scores[seed] = score

    seed_scores_by_year[year] = seed_scores

print("\nSEED SCORES SAMPLE:\n")
print(seed_scores_by_year["2010"])

years = sorted(seed_scores_by_year.keys())
#----------------------------------------------------------- inter graphs

reasons = {
    "2013": "Nirbhaya Case",
    "2015": "Beti Bachao",
    "2018": "POSHAN",
    "2020": "COVID",
    "2022": "Post-COVID"
}
for seed in seeds.keys():
    scores = [seed_scores_by_year[year][seed] for year in years]

    plt.plot(years, scores, label=seed)
for seed in seeds.keys():
    scores = [seed_scores_by_year[year][seed] for year in years]

    for i in range(1, len(scores)):
        if scores[i] > scores[i-1] * 1.5:
            year = years[i]
            value = scores[i]

            # Add annotation
            label = reasons.get(year, "")
            plt.text(year, value, label, fontsize=8, ha='center')

plt.xlabel("Year")
plt.ylabel("Score")
plt.title("Policy Trends Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

print("\nPEAK YEARS:\n")

for seed in seeds.keys():
    print(f"\n{seed.upper()}:")

    years = sorted(seed_scores_by_year.keys(), key=int)
    scores = [seed_scores_by_year[year][seed] for year in years]

    for i in range(1, len(scores)):
        if scores[i] > scores[i-1] * 1.5:   # spike condition
            print(f"Spike in {years[i]} (Score: {scores[i]})")