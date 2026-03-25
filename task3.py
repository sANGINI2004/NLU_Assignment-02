
words_to_check = ["research", "student", "phd", "exam"]

print("\n" + "="*50)
print("NEAREST NEIGHBORS (CBOW)")
print("="*50)

for word in words_to_check:
    if word in cbow_model.wv:
        print(f"\nWord: {word}")
        print(cbow_model.wv.most_similar(word, topn=5))
    else:
        print(f"\nWord: {word} not in vocabulary")


print("\n" + "="*50)
print("NEAREST NEIGHBORS (SKIP-GRAM)")
print("="*50)

for word in words_to_check:
    if word in skipgram_model.wv:
        print(f"\nWord: {word}")
        print(skipgram_model.wv.most_similar(word, topn=5))
    else:
        print(f"\nWord: {word} not in vocabulary")


# -------- ANALOGY TASK --------

print("\n" + "="*50)
print("ANALOGY RESULTS (CBOW)")
print("="*50)

# UG : BTech :: PG : ?
try:
    print("UG : BTech :: PG : ?")
    print(cbow_model.wv.most_similar(positive=["pg", "btech"], negative=["ug"], topn=5))
except:
    print("Analogy failed for CBOW")


# Student : Course :: Teacher : ?
try:
    print("\nStudent : Course :: Teacher : ?")
    print(cbow_model.wv.most_similar(positive=["teacher", "course"], negative=["student"], topn=5))
except:
    print("Analogy failed for CBOW")


# PhD : Research :: BTech : ?
try:
    print("\nPhD : Research :: BTech : ?")
    print(cbow_model.wv.most_similar(positive=["btech", "research"], negative=["phd"], topn=5))
except:
    print("Analogy failed for CBOW")


print("\n" + "="*50)
print("ANALOGY RESULTS (SKIP-GRAM)")
print("="*50)

# same analogies for skip-gram
try:
    print("UG : BTech :: PG : ?")
    print(skipgram_model.wv.most_similar(positive=["pg", "btech"], negative=["ug"], topn=5))
except:
    print("Analogy failed for Skip-gram")

try:
    print("\nStudent : Course :: Teacher : ?")
    print(skipgram_model.wv.most_similar(positive=["teacher", "course"], negative=["student"], topn=5))
except:
    print("Analogy failed for Skip-gram")

try:
    print("\nPhD : Research :: BTech : ?")
    print(skipgram_model.wv.most_similar(positive=["btech", "research"], negative=["phd"], topn=5))
except:
    print("Analogy failed for Skip-gram")
