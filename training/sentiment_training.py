import os
import warnings
import numpy as np  # linear algebra
import pandas as pd
from preprocessing import preprocessing
from Word2Vec import embedding
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from models import dl_models

os.environ["CUDA_VISIBLE_DEVICES"] = ""

warnings.filterwarnings('ignore')

#
model_name = 'blstm_sent'
tokenizer_name = model_name + '_tokenizer.pickle'

# Hyper-parameters
BATCH_SIZE = 64
EMBEDDING_SIZE = 100
SENTENCE_LENGTH = 12
OPTIMIZER = 'adam'
EPOCHS = 100

# Paths
data_path = '..\\data\\reviews.csv'
saved_model_path = '..\\models\\saved_models\\' + model_name + '.h5'
tokenizer_path = '..\\tokenizers\\' + tokenizer_name
word2vec_path = '..\\Word2Vec\\embeddings\\word2vec.txt'
acc_graph_path = '..\\graphs\\' + model_name + '_accuracy.png'
loss_graph_path = '..\\graphs\\' + model_name + 'loss.png'
metrics_path = '..\\reports\\' + model_name + "_reports.txt"

# Load data
print('fetching data!')
df = pd.read_csv(data_path)

# Initialize tokenizer
tokenizer = preprocessing.init_tokenizer()

# Show data
print(df.head())

# Get reviews and labels 0=neg, 1=pos
reviews = df['review'].to_list()
labels = df['sentiment'].to_list()

# Clean reviews
print('cleaning reviews!')
processed_reviews = preprocessing.clean_reviews(reviews)

# Get max length
if model_name == 'han_sent':
    max_length = 620
else:
    max_length = preprocessing.get_max_length(processed_reviews)

# Tokenize reviews
print('tokenizing reviews!')
sequences = preprocessing.tokenize_reviews(processed_reviews, tokenizer, True, tokenizer_path)

# Pad reviews
print('padding reviews!')
if (model_name == 'han_sent'):
    padded_reviews = preprocessing.pad_reviews(sequences, max_length * SENTENCE_LENGTH, 'post')
else:
    padded_reviews = preprocessing.pad_reviews(sequences, max_length, 'post')

# Create embedding dictionary
print('creating dictionary and embedding matrix!')
dictionary = embedding.create_embedding_dict(word2vec_path)
matrix = embedding.create_embedding_matrix(tokenizer, dictionary, EMBEDDING_SIZE)

# Convert lables to nparrays
review_labels = np.asarray(labels)

# Split data 80:20
print('splitting data!')
X_train, X_test, y_train, y_test = train_test_split(padded_reviews, review_labels, test_size=0.2)

# Reshape input tensor for han
if model_name == 'han_sent':
    X_train = preprocessing.reshape_inputs(X_train, SENTENCE_LENGTH, max_length)
    X_test = preprocessing.reshape_inputs(X_test, SENTENCE_LENGTH, max_length)

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Number of unique words
num_words = len(tokenizer.word_index) + 1

# create models
print('initialize model!')
model = dl_models.blstm(num_words, EMBEDDING_SIZE, matrix, max_length)
# model = dl_models.yoon_cnn(num_words, EMBEDDING_SIZE, matrix, max_length)
# model = dl_models.shallow_cnn(num_words, EMBEDDING_SIZE, matrix, max_length)
# model = dl_models.han(num_words, EMBEDDING_SIZE, matrix, max_length, SENTENCE_LENGTH)

# Create callbacks
print('model initialized!')

callbacks = dl_models.create_callbacks(saved_model_path, "val_loss", 'min', 2)

###############################################################################################
model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
model.summary()

print('training started!')
train_hist = model.fit(X_train,
                       y_train,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       validation_split=0.2,
                       callbacks=callbacks,
                       shuffle=True,
                       verbose=1)

# Draw training graphs
print('saving graphs')
dl_models.acc_loss_graphs_to_file(model_name, train_hist, ['train', 'val'], 'upper left', loss_graph_path,
                                  acc_graph_path)

# Evaluate models
print('evaluating models')
eval_report = model.evaluate(X_test, y_test, verbose=2, batch_size=BATCH_SIZE)
predictions = model.predict(X_test, verbose=2)
predictions = np.argmax(predictions, axis=1)

print("Eval loss: %.4f" % eval_report[0])
print("Model Accuracy: %.4f" % eval_report[1])

dl_models.metrics_to_file(model_name + '_results', metrics_path, y_test, predictions, ['neg', 'pos'], eval_report,
                          BATCH_SIZE, OPTIMIZER)

print('done!')
