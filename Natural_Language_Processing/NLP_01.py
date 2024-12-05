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





#or 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

def pre_process(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower()  not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

text = "Neural Language Processing(NLP) is a field of artificial intellignece that focuses on"
preprocessed_text = pre_process(text)
print("Preprocessed Text is: ", preprocessed_text)

