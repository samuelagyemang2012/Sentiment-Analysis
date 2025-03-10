import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, LSTM, GRU, Bidirectional, Dropout, Input, \
    Reshape, Conv2D, Conv1D, GlobalMaxPooling1D, MaxPooling2D, MaxPooling1D, Concatenate, TimeDistributed, ReLU
from tensorflow.keras.initializers import Constant
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, \
    accuracy_score
from tensorflow.keras import regularizers
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint
from models.attention_helper import Attention
from keras.layers import CuDNNLSTM, CuDNNGRU


# Defines all models for training and other functions

def blstm(num_words, embedding_size, embedding_matrix, input_length):
    model = Sequential()
    embedding_layer = Embedding(num_words,
                                embedding_size,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=input_length,
                                trainable=True)
    model.add(embedding_layer)
    model.add(Bidirectional(CuDNNLSTM(300)))
    model.add(Dense(300))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Dense(300))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model


# CNN model implemented by Kim Yoon
# Paper: Convolutional Neural Networks for Sentence Classification
# url: https://www.aclweb.org/anthology/D14-1181/
def yoon_cnn(num_words, embedding_size, embedding_matrix, input_length):
    # Hyperparameters
    filter_sizes = [3, 4, 5]  # defined convs regions
    num_filters = 100  # num_filters per conv region
    drop = 0.5

    embedding_layer = Embedding(num_words,
                                embedding_size,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=input_length,
                                trainable=True)

    inputs = Input(shape=(input_length,), dtype='int32')
    embedding = embedding_layer(inputs)
    reshape = Reshape((input_length, embedding_size, 1))(embedding)

    conv_0 = Conv2D(num_filters, (filter_sizes[0], embedding_size), activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_1 = Conv2D(num_filters, (filter_sizes[1], embedding_size), activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_2 = Conv2D(num_filters, (filter_sizes[2], embedding_size), activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)

    maxpool_0 = MaxPooling2D((input_length - filter_sizes[0] + 1, 1), strides=(1, 1))(conv_0)
    maxpool_1 = MaxPooling2D((input_length - filter_sizes[1] + 1, 1), strides=(1, 1))(conv_1)
    maxpool_2 = MaxPooling2D((input_length - filter_sizes[2] + 1, 1), strides=(1, 1))(conv_2)

    merged_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(merged_tensor)
    dropout1 = Dropout(drop)(flatten)
    output = Dense(units=2, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(dropout1)

    # this creates a model that includes
    model = Model(inputs, output)
    return model


def shallow_cnn(num_words, embedding_size, embedding_matrix, input_length):
    model = Sequential()

    model.add(Embedding(num_words,
                        embedding_size,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=input_length,
                        trainable=True))
    model.add(Dropout(0.2))

    model.add(Conv1D(300,
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(300))
    model.add(Dropout(0.2))
    model.add(ReLU())
    model.add(Dense(2, activation='softmax'))

    return model


# Hierarchical model implemented by Yang et al
# Paper: Hierarchical Attention Networks for Document Classification
# url: https://www.aclweb.org/anthology/N16-1174.pdf
# **
# max length word = input_length, max features = num_words
def han(num_words, embedding_size, embedding_matrix, input_length, sentence_length):
    input_word = Input(shape=(input_length,))
    x_word = Embedding(num_words,
                       embedding_size,
                       embeddings_initializer=Constant(embedding_matrix),
                       input_length=input_length,
                       trainable=True)(input_word)

    x_word = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x_word)
    x_word = Attention(input_length)(x_word)
    model_word = Model(input_word, x_word)

    # Sentence part
    input = Input(shape=(sentence_length, input_length,))
    x_sentence = TimeDistributed(model_word)(input)
    x_sentence = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x_sentence)
    x_sentence = Attention(sentence_length)(x_sentence)

    output = Dense(2, activation='softmax')(x_sentence)
    model = Model(inputs=input, outputs=output)
    return model


def plot_graphs(title, train_hist, val_hist, x_label, y_label, legend, loc, path):
    plt.plot(train_hist)
    plt.plot(val_hist)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend(legend, loc=loc)
    plt.savefig(path)
    plt.clf()

    # plt.show()


def show_confusion_matrix(y_test, preds):
    res = confusion_matrix(y_test.argmax(axis=1), preds)
    return "*Confusion Matrix*" + "\n" + np.array_str(res) + "\n"


def show_classification_report(y_test, preds, names):
    res = classification_report(y_test.argmax(axis=1), preds, target_names=names)
    return "*Classification Report*" + "\n" + res + "\n"


def get_metrics(eval_report, y_test, preds):
    data = "Accuracy: " + str(eval_report[1]) + "\n"

    tn, fp, fn, tp = confusion_matrix(y_test.argmax(axis=1), preds).ravel()

    data += 'False positive rate: ' + str(fp / (fp + tn)) + "\n"
    data += 'False negative rate: ' + str(fn / (fn + tp)) + "\n"

    recall = tp / (tp + fn)
    data += 'Recall: ' + str(recall) + "\n"

    precision = tp / (tp + fp)
    data += 'Precision: ' + str(precision) + "\n"

    f1_score = ((2 * precision * recall) / (precision + recall))
    data += 'F1 score: ' + str(f1_score) + "\n"

    return data


def get_fpr(y_test, preds):
    tn, fp, fn, tp = confusion_matrix(y_test.argmax(axis=1), preds).ravel()
    fpr = (fp / (fp + tn))
    return fpr


def get_fnr(y_test, preds):
    tn, fp, fn, tp = confusion_matrix(y_test.argmax(axis=1), preds).ravel()
    fnr = (fn / (fn + tp))
    return fnr


def metrics_to_file(file_name, path, y_test, preds, classes, eval_report, batch_size, optimizer):
    data = file_name + "\n"
    data += "*HYPER-PARAMETERS*" + "\n"
    data += "BATCH_SIZE: " + str(batch_size) + ", OPTIMIZER: " + optimizer + "\n"
    data += show_confusion_matrix(y_test, preds)
    data += show_classification_report(y_test, preds, classes)
    data += get_metrics(eval_report, y_test, preds)
    write_to_file(data, path)


def acc_loss_graphs_to_file(model_name, history, legend, legend_loc, loss_path, acc_path):
    loss_title = model_name + " Loss Graph"
    acc_title = model_name + " Accuracy Graph"

    train_acc = history.history['acc']
    val_acc = history.history['val_acc']

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plot_graphs(loss_title, train_loss, val_loss, "epochs", "loss", legend, legend_loc, loss_path)
    plot_graphs(acc_title, train_acc, val_acc, "epochs", "accuracy", legend, legend_loc, acc_path)


def write_to_file(data, path):
    f = open(path, "w")
    f.write(data)
    f.close()


def create_callbacks(best_model_path, monitor, mode, patience):
    es = EarlyStopping(monitor=monitor, mode=mode, verbose=1, patience=patience)
    mc = ModelCheckpoint(best_model_path, monitor=monitor, mode=mode, verbose=1, save_best_only=True)
    callbacks = [es, mc]
    return callbacks


def append_metrics(y_test, y_preds, accs, fnrs, fprs, precs, recalls, f1s, epchs, model):
    acc = accuracy_score(y_test.argmax(axis=1), y_preds)
    if acc != 'nan':
        accs.append(acc)
    else:
        accs.append(0)

    fnr = get_fnr(y_test, y_preds)
    if fnr != 'nan':
        fnrs.append(fnr)
    else:
        fnrs.append(0)

    fpr = get_fpr(y_test, y_preds)
    if fpr != 'nan':
        fprs.append(fpr)
    else:
        fprs.append(0)

    prec = precision_score(y_test.argmax(axis=1), y_preds, average='binary')
    if prec != 'nan':
        precs.append(prec)
    else:
        precs.append(0)

    rec = recall_score(y_test.argmax(axis=1), y_preds, average='binary')
    if rec != 'nan':
        recalls.append(rec)
    else:
        recalls.append(0)

    f1 = f1_score(y_test.argmax(axis=1), y_preds, average='binary')
    if f1 != 'nan':
        f1s.append(f1)
    else:
        f1s.append(0)

    ep = len(model.history['loss'])
    # print('EP: ', str(ep))
    if ep != 'nan':
        epchs.append(ep)
    else:
        epchs.append(0)


def get_averages(accs, fnrs, fprs, precs, recalls, f1s, epchs):
    avg_acc = mean(accs)
    print('Accuracy: ', str(avg_acc))

    avg_fnr = mean(fnrs)
    print('FNR: ', str(avg_fnr))

    avg_fpr = mean(fprs)
    print('FPR: ', str(avg_fpr))

    avg_prec = mean(precs)
    print('Precision: ', str(avg_prec))

    avg_rec = mean(recalls)
    print('Recall: ', str(avg_rec))

    avg_f1 = mean(f1s)
    print('F1 score: ', str(avg_f1))

    avg_eps = mean(epchs)
    print('Epochs: ', str(avg_eps))


def clear_lists(accs, fnrs, fprs, precs, recalls, f1s, epchs):
    accs.clear()
    fnrs.clear()
    fprs.clear()
    precs.clear()
    recalls.clear()
    f1s.clear()
    epchs.clear()
