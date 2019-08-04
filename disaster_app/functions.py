
from sklearn.base import BaseEstimator, TransformerMixin

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
