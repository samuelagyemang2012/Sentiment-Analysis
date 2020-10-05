import pickle
from keras.models import load_model


def load_dl_model(path, custom_objs):
    # if dict is empty
    if bool(custom_objs):
        return load_model(path, custom_objects=custom_objs)
    else:
        return load_model(path)


def get_model_input_shape(model):
    return model.input.shape


def load_tokenizer(path):
    with open(path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def tokenize_reviews(reviews, tokenizer):
    sequences = tokenizer.texts_to_sequences(reviews)
    return sequences


def display_results(reviews, preds):
    pos = 0
    n_preds = preds.argmax(axis=1)
    for i in range(0, len(reviews)):
        print(reviews[i])
        if n_preds[i] == 0:
            print('*negative*')
            print(' Confidence: ' + str(round((preds[i].max() * 100), 2)) + '%')
        else:
            print('*positive*')
            print(' Confidence: ' + str(round((preds[i].max() * 100), 2)) + '%')
            pos += 1

        print('------------------------------------')

    pos_percent = (pos / len(reviews)) * 100
    print('#Total reviews: ', len(reviews))
    print('#Positive reviews: ', pos)
    print("#Total positives:", round(pos_percent, 2))


def predict(model, sequences, user_reviews):
    preds = model.predict(sequences)
    display_results(user_reviews, preds)


def predict_from_csv(input_path, output_path):
    return 1
