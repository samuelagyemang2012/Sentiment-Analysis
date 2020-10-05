import os
import pandas as pd
import numpy as np
import gensim
from gensim.models import KeyedVectors
from nltk import word_tokenize
from gensim.models.word2vec import Word2Vec


def get_embeddings(words, embedding_size, iteration, path):
    w2v = gensim.models.Word2Vec(sentences=words, min_count=1, size=embedding_size, window=5, workers=10,
                                 iter=iteration)
    w2v.wv.save_word2vec_format(path, binary=False)
    print('embeddings saved!')


def load_embeddings(path):
    embeddings = open(os.path.join('', path), encoding='utf-8')
    return embeddings


def load_w2v_model(path, is_binary):
    model = KeyedVectors.load_word2vec_format(path, binary=is_binary)
    return model


def create_embedding_dict(path):
    embeddings = load_embeddings(path)
    dictionary = {}

    for line in embeddings:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:])
        dictionary[word] = vectors

    embeddings.close()
    print('dictionary created!')
    return dictionary


def create_embedding_matrix(tokenizer, dictionary, embedding_size):
    word_index = tokenizer.word_index
    num_words = len(word_index) + 1

    embedding_matrix = np.zeros((num_words, embedding_size))

    for word, num in word_index.items():

        if num > num_words:
            continue
        embedding_vector = dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[num] = embedding_vector
        else:
            embedding_matrix[num] = np.random.randn(embedding_size)

    print('embedding matrix created')
    return embedding_matrix
