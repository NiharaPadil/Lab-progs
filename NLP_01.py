import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.data import find
def download_nltk_data():
    try:
        find('corpora/stopwords.zip')
        find('tokenizers/punkt.zip')
    except LookupErroe:
        nltk.download('stopwords')
        nltk.download('punkt')
def tokenize(text):
    return word_tokenize(text)
def filter_text(tokens):
    return [re.sub(r'[^a-zA-Z0-9]', '',token) for token in token if token]
def validate_script(tokens):
    return[token for token in tokens if re.match(r'^[a-zA-Z]+$',token)]
def remove_stop_words(tokens):
    stop_words=set(stopwords.word('english'))
    return [token for token in tokens if token.lower()not in stop_words]
def stem_tokens(tokens):
    stemmer =PorterStemmer()
    return [stemmer.stem(token) for token in tokens]
def preprocess_text(text):
    download_nltk_data()
    tokens= tokenize(text)
    tokens=filter_text(tokens)




#------------




import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.data import find

def download_nltk_data():
    try:
        find('corpora/stopwords.zip')
        find('tokenizers/punkt.zip')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')

def tokenize(text):
    return word_tokenize(text)

def filter_text(tokens):
    return [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens if token]

def validate_script(tokens):
    return [token for token in tokens if re.match(r'^[a-zA-Z]+$', token)]

def remove_stop_words(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token.lower() not in stop_words]

def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text):
    download_nltk_data()
    tokens = tokenize(text)
    tokens = filter_text(tokens)
    tokens = validate_script(tokens)
    tokens = remove_stop_words(tokens)
    tokens = stem_tokens(tokens)
    return tokens

if __name__ == "__main__":
    sample_text = "Neural Language Processing (NLP) is a field of artificial intelligence that focuses on"
   
    processed_tokens = preprocess_text(sample_text)
    print("Processed Tokens:", processed_tokens)
