import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist, MLEProbDist
from nltk.tokenize import word_tokenize  # Corrected import

nltk.download('punkt')

sentence = "I Love programming in Python and I enjoy learning languages"

# Tokenize and convert to lowercase
tokens = word_tokenize(sentence.lower())

# Generate n-grams
unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

# Calculate frequency distributions
unigram_freq = FreqDist(unigrams)
bigram_freq = FreqDist(bigrams)
trigram_freq = FreqDist(trigrams)

# Calculate maximum likelihood estimators for probabilities
unigram_prob_dist = MLEProbDist(unigram_freq)
bigram_prob_dist = MLEProbDist(bigram_freq)
trigram_prob_dist = MLEProbDist(trigram_freq)

def get_ngram_prob(ngram, prob_dist):
    if not isinstance(ngram, tuple):
        ngram = tuple(ngram)
    return prob_dist.prob(ngram)

# Print unigram probabilities
print("\nUnigram Probabilities")
for unigram in unigram_freq:
    print(f"{unigram[0]}: {get_ngram_prob(unigram, unigram_prob_dist)}")

# Print bigram probabilities
print("\nBigram Probabilities")
for bigram in bigram_freq:
    print(f"{bigram[0]}: {get_ngram_prob(bigram, bigram_prob_dist)}")

# Print trigram probabilities
print("\nTrigram Probabilities")
for trigram in trigram_freq:
    print(f"{trigram[0]}: {get_ngram_prob(trigram, trigram_prob_dist)}")
