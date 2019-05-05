import time
import pprint
import pickle
import numpy as np
import cnn_utilities
from numpy import array
from math import sqrt, log
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.initializers import Constant
from keras.layers import Input, Dense, GlobalMaxPooling1D, Reshape, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Dropout
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, mean_squared_error, \
    r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight

start_runtime = time.time()
# set run number
n = 5

# select for raw text sdps or POS sdps
# pos = False
pos = True

# set balance to True to exclude "USAGE" instances
# balance = False
# balance = True

# set balance to "TOP" for instances count > 100,
#                "LOW" for instances count < 100
#                "ALL" for all instances
balance = "ALL"
# balance = "TOP"
# balance = "LOW"
TOP = ["USAGE", "MODEL-FEATURE", "PART_WHOLE"]
LOW = ["RESULT", 'COMPARE', 'TOPIC']

# set model architecture to use
# model_type = "rand"  # one channel, random initialization
# model_type = "w-em"  # one channel, word embedding initialization
model_type = "twoC"  # two channels, one random and one word embedding initialization

# set l2 regularizer
l_two = 3

# set class weight mu
# mu = 0.35

print("Classifying Relations, input is pos={} and balance={}".format(pos, balance))
print("Model architecture: ".format(model_type))
logfile = open("experiment_logs/test{}_pos={}_balance={}_modelType={}_l2={}.txt"
               .format(n, pos, balance, model_type, l_two), 'w')

# getting the feature files
all_paths = []  # list of all paths
all_rels = []   # list of all relation labels

# training features
path = 'clean/train_data/'
# just text relations
filename = 'data_3.0/1.1.features.txt'
if pos:
    filename = 'data_3.0/1.1.features_pos.txt'
train_paths = []   # list of train input paths
train_rels = []    # list of train relation labels
label_counts = {}  # dictionary of label, count
MAX_PATH_LENGTH = 0
with open(path + filename) as f:
    print("Loading training features from {}".format(filename))
    for line in f:
        line = line.strip()
        line = line.split()
        rel = line[0]
        if balance != "ALL":
            if balance == "TOP" and rel not in TOP:
                continue
            if balance == "LOW" and rel not in LOW:
                continue
        train_rels.append(rel)
        all_rels.append(rel)
        path = line[1:]
        if len(path) > MAX_PATH_LENGTH:
            MAX_PATH_LENGTH = len(path)
        path = ' '.join(path)
        train_paths.append(path)
        if rel not in label_counts.keys():
            label_counts[rel] = 0
        label_counts[rel] += 1
        all_paths.append(path)

# print(all_paths[0])
# print(len(paths[0].split()))
print("longest input path length:", MAX_PATH_LENGTH)

# getting the test feature files
path = 'clean/test_data/'
# just text relations
filename = 'data_3.0/1.1.test.features.txt'
if pos:
    filename = 'data_3.0/1.1.test.features_pos.txt'
test_paths = []  # list of input paths
test_rels = []  # list of relation labels
test_label_counts = {}  # dictionary of label, count in test set
with open(path + filename) as f:
    print("Loading test features from {}".format(filename))
    for line in f:
        line = line.strip()
        line = line.split()
        rel = line[0]
        if balance != "ALL":
            if balance == "TOP" and rel not in TOP:
                continue
            if balance == "LOW" and rel not in LOW:
                continue
        test_rels.append(rel)
        all_rels.append(rel)
        path = line[1:]
        if len(path) > MAX_PATH_LENGTH:
            MAX_PATH_LENGTH = len(path)
        path = ' '.join(path)
        test_paths.append(path)
        if rel not in test_label_counts.keys():
            test_label_counts[rel] = 0
        test_label_counts[rel] += 1
        all_paths.append(path)

# print(test_rels[0])
# print(len(paths[0].split()))
print("longest input path length:", MAX_PATH_LENGTH)

# vectorize text features into 2D integer tensor
MAX_SEQ_LEN = 100
# tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer = Tokenizer(filters='!"#$%&()*+,./:;=?@[\\]^_`{|}~\t\n')  # modified to include --> and <--
tokenizer.fit_on_texts(all_paths)
word_index = tokenizer.word_index
print("Found {} unique tokens".format(len(word_index)))
logfile.write("{} unique tokens\n".format(len(word_index)))

train_sequences = tokenizer.texts_to_sequences(train_paths)
train_data = pad_sequences(train_sequences, maxlen=MAX_SEQ_LEN, padding='post')
# print(len(train_sequences[0]))
# print(train_sequences[0])

test_sequences = tokenizer.texts_to_sequences(test_paths)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQ_LEN, padding='post')
# print(len(test_sequences[0]))
# print(test_sequences[0])

le = LabelEncoder()
le.fit(all_rels)
train_labels = le.transform(train_rels)
test_labels = le.transform(test_rels)
train_labels = to_categorical(np.asarray(train_labels))
test_labels = to_categorical(np.asarray(test_labels))
print("Shape of train data tensor: ", train_data.shape)
print("Shape of train label tensor: ", train_labels.shape)
print("Shape of test data tensor: ", test_data.shape)
print("Shape of test label tensor: ", test_labels.shape)

# splitting into training and validation
VALIDATION_SPLIT = 0.2
indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)
data = train_data[indices]
labels = train_labels[indices]
# sample_weights = sample_weights[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
sss = StratifiedShuffleSplit(test_size=VALIDATION_SPLIT)
for train_ind, test_ind in sss.split(train_data, train_labels):
    x_train, x_val = train_data[train_ind], train_data[test_ind]
    y_train, y_val = train_labels[train_ind], train_labels[test_ind]

# for simple validation splitting
# x_train = data[:-num_validation_samples]
# y_train = labels[:-num_validation_samples]
# y_train = to_categorical(np.asarray(y_train))
# x_val = data[-num_validation_samples:]
# y_val = labels[-num_validation_samples:]
# y_val = to_categorical(np.asarray(y_val))
# print(y_val)

# getting sample weights
orig_y = np.argmax(y_train, axis=1)
orig_y = le.inverse_transform(orig_y)
# print(orig_y)
total_labels = len(train_rels)
sample_weights = [1 - float(label_counts[orig_y[i]]) / total_labels for i in range(len(orig_y))]

# print(sample_weights)

# getting class weights
# class_weights = cnn_utilities.get_class_weights(label_counts, le, mu=mu)
y_ints = [y.argmax() for y in y_train]
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y_ints),
                                                  y_ints)
print(class_weights)

# below is just a renaming of test_data and test_labels, for ease of reading
x_test = test_data
y_test = test_labels
print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of x_val: ", x_val.shape)
print("Shape of y_val: ", y_val.shape)
print("Shape of x_test: ", x_test.shape)
print("Shape of y_test: ", y_test.shape)

# loading the word embeddings
if pos:
    # filename = 'pos_embeddings_index.pkl'
    filename = 'combo_index.pkl'
else:
    filename = 'embeddings_index.pkl'
print("loading embedding file", filename)
em_file = open(filename, 'rb')
embeddings_index = pickle.load(em_file)
em_file.close()

print("Found {} word vectors.".format(len(embeddings_index)))
logfile.write("{} word vectors found in {}\n".format(len(embeddings_index),
                                                     filename))
# creating the embedding matrix
EMBEDDING_DIM = 300
num_words = len(word_index) + 1
print(len(word_index)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    # print(word)
    word = word.upper()
    # print(i)
    if i > num_words - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
nonzero_elems = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print("word coverage:", nonzero_elems / num_words)
print("embedding_matrix shape:", embedding_matrix.shape)
logfile.write("word coverage: {}\n".format(nonzero_elems / num_words))

# creating the random initialization embedding matrix
rand_embedding_matrix = np.random.rand(num_words, EMBEDDING_DIM)

# channel 1 - random initialization embedding
inputs1 = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
embed1 = Embedding(rand_embedding_matrix.shape[0],
                   rand_embedding_matrix.shape[1],
                   input_length=MAX_SEQ_LEN,
                   weights=[rand_embedding_matrix],
                   embeddings_initializer=Constant(rand_embedding_matrix),
                   trainable=True,
                   name="Embed_rand")(inputs1)
branches1 = cnn_utilities.get_conv_and_pool(embed1, "rand",
                                            kr=regularizers.l2(l_two),
                                            feature_maps=128)
channel1 = concatenate(branches1, axis=-1)

# channel 2 - pre-trained word vectors from NLPL
inputs2 = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
embed2 = Embedding(len(word_index) + 1,
                   EMBEDDING_DIM,
                   weights=[embedding_matrix],
                   embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_SEQ_LEN,
                   trainable=True)(inputs2)
branches2 = cnn_utilities.get_conv_and_pool(embed2, "wiki",
                                            kr=regularizers.l2(l_two),
                                            feature_maps=128)
channel2 = concatenate(branches2, axis=-1)

if model_type == "rand":
    output = Dense(128, activation='relu',
                   # kernel_regularizer=regularizers.l2(l_two)
                   )(channel1)
elif model_type == "w-em":
    output = Dense(128, activation='relu',
                   # kernel_regularizer=regularizers.l2(l_two)
                   )(channel2)
else:  # model_type == "twoC":
    merged = concatenate([channel2, channel1])
    output = Dense(128, activation='relu',
                   # kernel_regularizer=regularizers.l2(l_two)
                   )(merged)

# output layer
if balance == "TOP" or balance == "LOW":
    output = Dense(3, activation='softmax',
                   # kernel_regularizer=regularizers.l2(l_two)
                   )(output)
else:
    output = Dense(6, activation='softmax',
                   # kernel_regularizer=regularizers.l2(l_two)
                   )(output)

if model_type == "rand":
    model = Model(inputs=inputs1, outputs=output)
elif model_type == "w-em":
    model = Model(inputs=inputs2, outputs=output)
else:  # model_type == "twoC":
    model = Model(inputs=[inputs2, inputs1], outputs=output)

model.compile(
              loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc', 'mse', 'cosine'],)
model.summary()
print("model compiled successfully")

print("Training the model...")

start_time = time.time()
if model_type == "twoC":
    history = model.fit(
                    [x_train, x_train],
                    array(y_train),
                    batch_size=128,
                    epochs=10,
                    # sample_weight=array(sample_weights),
                    class_weight=class_weights,
                    validation_data=([x_val, x_val], array(y_val)),
                    verbose=2)
else:
    history = model.fit(
                    x_train,
                    y_train,
                    batch_size=128,
                    epochs=10,
                    # sample_weight=array(sample_weights),
                    class_weight=class_weights,
                    validation_data=(x_val, y_val),
                    verbose=2)
end_time = np.round(time.time() - start_time, 2)
print("model fitted on x_train, y_train")
print("Training time:", end_time)
logfile.write("\nTraining time: {}s\n".format(end_time))
print()

print("Evaluating model...")
start_time = time.time()
if model_type == "twoC":
    score, acc, mse_me, cosine = model.evaluate([x_val, x_val], array(y_val), verbose=1)
else:
    score, acc, mse_me, cosine = model.evaluate(x_val, y_val, verbose=1)
end_time = np.round(time.time() - start_time, 2)
print("Eval time:", end_time)
print("score:", score)
print("acc:", acc)
print("model mse:", mse_me)
print("cosine:", cosine)
logfile.write("\nEval time: {}s\n".format(end_time))
logfile.write("score: {}\nacc: {}\nmodel mse: {}\ncosine: {}\n"
              .format(score, acc, mse_me, cosine))
print()

print("Evaluating class predictions on x_train...")
label_names = le.classes_
print(label_names)
start_time = time.time()
if model_type == "twoC":
    y_pred = model.predict([x_train, x_train], 128, verbose=2)
else:
    y_pred = model.predict(x_train, 128, verbose=2)
end_time = np.round(time.time() - start_time, 2)
print("Predict time:", end_time)
print(classification_report(y_train.argmax(axis=1),
                            y_pred.argmax(axis=1),
                            target_names=label_names))
test_acc = accuracy_score(y_train.argmax(axis=1),
                          y_pred.argmax(axis=1))
print("Accuracy:", test_acc)
mse = mean_squared_error(y_train, y_pred)
print("Mean Squared Error:", mse)
print("RMSE:", sqrt(mse))
r2 = r2_score(y_train, y_pred)
print("r2:", r2)
logfile.write("\nClass predictions on x_val\n")
logfile.write("Predict time: {}\nAccuracy: {}\nMean Squared Error: {}\nRMSE: {}\nR2: {}\n"
              .format(end_time, test_acc, mse, sqrt(mse), r2))
logfile.write(classification_report(y_train.argmax(axis=1),
                                    y_pred.argmax(axis=1),
                                    target_names=label_names))

print("Evaluating class predictions on x_val...")
label_names = le.classes_
print(label_names)
start_time = time.time()
if model_type == "twoC":
    y_pred = model.predict([x_val, x_val], 128, verbose=2)
else:
    y_pred = model.predict(x_val, 128, verbose=2)
end_time = np.round(time.time() - start_time, 2)
print("Predict time:", end_time)
print(classification_report(y_val.argmax(axis=1),
                            y_pred.argmax(axis=1),
                            target_names=label_names))
test_acc = accuracy_score(y_val.argmax(axis=1),
                          y_pred.argmax(axis=1))
print("Accuracy:", test_acc)
mse = mean_squared_error(y_val, y_pred)
print("Mean Squared Error:", mse)
print("RMSE:", sqrt(mse))
r2 = r2_score(y_val, y_pred)
print("r2:", r2)
logfile.write("\nClass predictions on x_val\n")
logfile.write("Predict time: {}\nAccuracy: {}\nMean Squared Error: {}\nRMSE: {}\nR2: {}\n"
              .format(end_time, test_acc, mse, sqrt(mse), r2))
logfile.write(classification_report(y_val.argmax(axis=1),
                                    y_pred.argmax(axis=1),
                                    target_names=label_names))

print("Evaluating class predictions on x_test...")
label_names = le.classes_
print(label_names)
start_time = time.time()
if model_type == "twoC":
    y_pred = model.predict([x_test, x_test], 128, verbose=2)
else:
    y_pred = model.predict(x_test, 128, verbose=2)
end_time = np.round(time.time() - start_time, 2)
print("Predict time:", end_time)
print(classification_report(y_test.argmax(axis=1),
                            y_pred.argmax(axis=1),
                            target_names=label_names))
test_acc = accuracy_score(y_test.argmax(axis=1),
                          y_pred.argmax(axis=1))
print("Accuracy:", test_acc)
test_prec = precision_score(y_test.argmax(axis=1),
                            y_pred.argmax(axis=1),
                            average='weighted')
print("precision:", test_prec)
test_rec = recall_score(y_test.argmax(axis=1),
                        y_pred.argmax(axis=1),
                        average='weighted')
test_f1 = f1_score(y_test.argmax(axis=1),
                   y_pred.argmax(axis=1),
                   average='macro')
print("f1:", test_f1)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("RMSE:", sqrt(mse))
r2 = r2_score(y_test, y_pred)
print("r2:", r2)
logfile.write("\nClass predictions on x_test\n")
logfile.write("\nPredict time: {}\nAccuracy: {}\nPrecision: {}\nRecall: {}\nMacro F1: {}\nMean Squared Error: {}\nRMSE: {}\nR2: {}\n"
              .format(end_time, test_acc, test_prec, test_rec, test_f1, mse, sqrt(mse), r2))
logfile.write(classification_report(y_test.argmax(axis=1),
                                    y_pred.argmax(axis=1),
                                    target_names=label_names))

print("label counts in training set:")
# pprint.pprint(label_counts)
logfile.write("\nlabel counts in training set: {}\n".format(label_counts))
print("label counts in test set:")
# pprint.pprint(test_label_counts)
logfile.write("label counts in test set: {}\n".format(test_label_counts))

end_runtime = np.round(time.time() - start_runtime, 2)
logfile.write("\ntotal runtime: {}".format(end_runtime))

logfile.close()

