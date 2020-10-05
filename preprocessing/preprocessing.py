import nltk

# nltk.download()
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

import string
import pickle
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def init_wordnet():
    wnl = WordNetLemmatizer()
    print('wordnet initialized!')
    return wnl


def init_tokenizer():
    tokenizer = Tokenizer()
    print('tokenizer initialized!')
    return tokenizer


def clean_reviews(reviews):
    wnl = init_wordnet()
    processed_reviews = []

    for review in reviews:
        # remove html tags
        soup = BeautifulSoup(review)
        review = soup.get_text()
        # tokenize review: separate each review them into words
        # TODO try sentence tokenizer
        tokens = word_tokenize(review)
        # convert words to lower case
        tokens = [t.lower() for t in tokens]
        # remove punctuations
        table = str.maketrans('', '', string.punctuation)
        stripped = [t.translate(table) for t in tokens]
        # remove non alphabetic tokens
        words = [word for word in stripped if word.isalpha()]
        # filter stopwords
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        words = [wnl.lemmatize(w) for w in words]
        processed_reviews.append(words)

    print('reviews cleaned!')
    return processed_reviews


def save_tokenizer(path, tokenizer):
    with open(path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def tokenize_reviews(reviews, tokenizer, t, path):
    tokenizer.fit_on_texts(reviews)
    if t:
        save_tokenizer(path, tokenizer)
        print('tokenizer saved!')

    sequences = tokenizer.texts_to_sequences(reviews)
    return sequences


def pad_reviews(reviews, max_length, method):
    padded_reviews = pad_sequences(reviews, maxlen=max_length, padding=method)
    print('reviews padded!')
    return padded_reviews


def get_max_length(reviews):
    list_len = [len(i) for i in reviews]
    max_length = max(list_len)
    return max_length


def reshape_inputs(data, sentence_length, max_length):
    return data.reshape((len(data), sentence_length, max_length))
