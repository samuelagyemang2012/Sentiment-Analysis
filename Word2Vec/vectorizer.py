import pandas as pd
from Word2Vec import embedding
from preprocessing import preprocessing

data_path = '..\\data\\reviews.csv'
save_path = '..\\Word2Vec\\embeddings\\word2vec.txt'

EMBEDDING_SIZE = 100
ITER = 50

# Load data
print('fetching data!')
df = pd.read_csv(data_path)

# Get reviews
reviews = df['review'].to_list()

# Clean reviews
print('cleaning reviews!')
processed_reviews = preprocessing.clean_reviews(reviews)

# Generate embeddings
embedding.get_embeddings(processed_reviews, EMBEDDING_SIZE, ITER, save_path)
