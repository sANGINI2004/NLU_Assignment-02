
from nltk.tokenize import sent_tokenize

# we will store all sentences here (each sentence will be a list of tokens)
sentences = []

# going through each document we loaded earlier
for doc_name, text in raw_documents.items():
    
    # split document into sentences
    raw_sents = sent_tokenize(text)
    
    for sent in raw_sents:
        # preprocess each sentence (same function from task 1)
        tokens = preprocess(sent)
        
        # only keep meaningful sentences (avoid very small ones)
        if len(tokens) > 2:
            sentences.append(tokens)

# total sentences that will be used for training Word2Vec
print("Total sentences for training:", len(sentences))


from gensim.models import Word2Vec

# -------- CBOW MODEL --------
# CBOW predicts a word using surrounding context words
cbow_model = Word2Vec(
    sentences=sentences,   # input data (list of tokenized sentences)
    vector_size=100,       # embedding dimension
    window=5,              # context window size
    min_count=2,           # ignore words with freq < 2
    sg=0,                  # sg=0 means CBOW
    negative=5             # number of negative samples
)



# -------- SKIP-GRAM MODEL --------
# Skip-gram predicts surrounding words from a given word
skipgram_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    sg=1,                  # sg=1 means Skip-gram
    negative=5
)

print("Both models trained successfully")


# checking similarity results for word "student"
print("\nCBOW:")
print(cbow_model.wv.most_similar("student"))

print("\nSkip-gram:")
print(skipgram_model.wv.most_similar("student"))


# -------- EXPERIMENTS (as required in assignment) --------

# 1. Higher embedding dimension (to capture more semantic info)
cbow_200 = Word2Vec(
    sentences,
    vector_size=200,   # increased dimension
    window=5,
    sg=0,
    negative=5
)

# 2. Larger context window (captures broader context)
skipgram_w10 = Word2Vec(
    sentences,
    vector_size=100,
    window=10,   # increased window size
    sg=1,
    negative=5
)

# 3. More negative samples (better training but slower)
skipgram_neg10 = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    sg=1,
    negative=10   # increased negative sampling
)

# -------- COMPARISON OF EXPERIMENTS --------
# we will compare models using similarity results

test_word = "student"

print("\n" + "="*50)
print("EXPERIMENT RESULTS COMPARISON")
print("="*50)

# --- CBOW default ---
print("\nCBOW (vector_size=100, window=5, negative=5):")
print(cbow_model.wv.most_similar(test_word))

# --- CBOW higher dimension ---
print("\nCBOW (vector_size=200):")
print(cbow_200.wv.most_similar(test_word))


# --- Skip-gram default ---
print("\nSkip-gram (window=5, negative=5):")
print(skipgram_model.wv.most_similar(test_word))

# --- Skip-gram larger window ---
print("\nSkip-gram (window=10):")
print(skipgram_w10.wv.most_similar(test_word))

# --- Skip-gram more negative samples ---
print("\nSkip-gram (negative=10):")
print(skipgram_neg10.wv.most_similar(test_word))
