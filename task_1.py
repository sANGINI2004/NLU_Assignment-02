# Step 1: Imports

import re
import string
import os
from collections import Counter

import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# pdfminer is used to properly extract text from PDF documents

from pdfminer.high_level import extract_text as pdf_extract_text

# Download required NLTK data (tokenizer and stopwords)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# Step 2: Upload files in Colab

from google.colab import files
print("Please upload all 5 source files now...")
uploaded = files.upload()
print(f"\nUploaded files: {list(uploaded.keys())}")


# Step 3: Load and extract text from each file

# The tricky part here is that Regulations.txt and Curriculum-BTech-CSE.txt
# are PDF files saved with a .txt extension. If we read them normally
# (as plain text), we get binary garbage like "endobj", "flatedecode", etc.
# We detect this by checking if the file starts with the PDF magic bytes "%PDF".
# If it does, we use pdfminer to extract the actual readable text.

def load_file(filename, file_bytes):
    # Check if the file is actually a PDF by looking at its first few bytes
    # PDF files always start with the signature "%PDF"
    if file_bytes[:4] == b'%PDF':
        # Save the bytes temporarily to disk so pdfminer can read it
        temp_path = f"/tmp/{filename}"
        with open(temp_path, 'wb') as f:
            f.write(file_bytes)
        # Use pdfminer to extract clean text from the PDF
        text = pdf_extract_text(temp_path)
        print(f"  [PDF detected] Extracted text from: {filename}")
    else:
        # Regular text file, just decode it normally
        text = file_bytes.decode('utf-8', errors='ignore')
        print(f"  [Text file] Loaded: {filename}")
    return text

# Load all uploaded files and store their raw text
print("\nLoading files...")
raw_documents = {}
for filename, file_bytes in uploaded.items():
    raw_documents[filename] = load_file(filename, file_bytes)

print(f"\nSuccessfully loaded {len(raw_documents)} documents.")


# Step 4: Preprocessing

stop_words = set(stopwords.words('english'))

# Extra domain-specific words to remove -- these appeared as noise
# in our corpus and don't carry useful semantic meaning for the task
extra_noise_words = {
    'iit', 'jodhpur', 'page', 'copyright', 'reserved',
    'www', 'http', 'https', 'com', 'org', 'edu'
}

def preprocess(text):
    # Remove non-ASCII characters (removes Hindi text and other scripts)
    text = text.encode('ascii', errors='ignore').decode()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove common boilerplate patterns found in scraped web pages
    text = re.sub(r'A\+\s*A\s*A-', '', text)
    text = re.sub(r'Copyright.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Last Updated.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'View all.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'arrow_downward', '', text)

    # Lowercase all text
    text = text.lower()

    # Remove digits (page numbers, years, etc.)
    text = re.sub(r'\d+', '', text)

    # Remove all punctuation characters
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text into individual words
    tokens = word_tokenize(text)

    # Remove stopwords, noise words, and tokens shorter than 3 characters
    tokens = [
        t for t in tokens
        if t not in stop_words
        and t not in extra_noise_words
        and len(t) > 2
    ]

    return tokens


# Apply preprocessing to all documents
print("\nPreprocessing documents...")
tokenized_docs = {}
all_tokens = []

for name, text in raw_documents.items():
    tokens = preprocess(text)
    tokenized_docs[name] = tokens
    all_tokens.extend(tokens)
    print(f"  {name}: {len(tokens)} tokens after cleaning")


# Step 5: Dataset statistics

word_freq = Counter(all_tokens)

print("\n" + "="*50)
print("DATASET STATISTICS")
print("="*50)
print(f"  Total documents      : {len(tokenized_docs)}")
print(f"  Total tokens         : {len(all_tokens)}")
print(f"  Vocabulary size      : {len(set(all_tokens))}")
print(f"\n  Top 20 most frequent words:")
for word, freq in word_freq.most_common(20):
    print(f"    {word:25s} -> {freq}")


# Step 6: Save the cleaned corpus

# We save all tokens to a text file where each document occupies one line.
# Each line is a "sentence" from the model's perspective.

corpus_path = "cleaned_corpus.txt"
with open(corpus_path, "w") as f:
    for name, tokens in tokenized_docs.items():
        # Write all tokens for this document as a single space-separated line
        f.write(" ".join(tokens) + "\n")

print(f"\nCleaned corpus saved to: {corpus_path}")


# Step 7: Word Cloud

print("\nGenerating word cloud...")

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    max_words=150,
    colormap='viridis',
    # Use word frequencies directly so the cloud reflects actual counts
    relative_scaling=0.5
).generate_from_frequencies(word_freq)

plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud — IIT Jodhpur Corpus", fontsize=18, pad=20)
plt.tight_layout()
plt.savefig("wordcloud.png", dpi=150, bbox_inches='tight')
plt.show()

print("Word cloud saved as: wordcloud.png")


# Step 8: Download output files

print("\nDownloading output files...")
files.download("cleaned_corpus.txt")
files.download("wordcloud.png")

