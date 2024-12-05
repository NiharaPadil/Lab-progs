import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
def get_synonyms_antonyms(word):
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Add synonym
            synonyms.append(lemma.name())
            # Add antonym if it exists
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    
    return set(synonyms), set(antonyms)

# Example usage
word = input('Enter the word to get antonym and synonym:').strip()
synonyms, antonyms = get_synonyms_antonyms(word)

print(f"Synonyms of '{word}': {synonyms}")
print(f"Antonyms of '{word}': {antonyms}")
