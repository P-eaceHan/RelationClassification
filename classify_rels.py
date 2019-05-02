import time
import pickle
import numpy as np
from math import sqrt
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.initializers import Constant
from keras.layers import Input, Dense, GlobalMaxPooling1D, Reshape, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Dropout
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score
from keras.callbacks import TensorBoard


# getting the feature files
all_paths = []  # list of all paths
all_rels = []   # list of all relation labels

# training features
path = 'clean/train_data/'
# just text relations
# filename = 'data_3.0/1.1.features.txt'
# POS relations
filename = 'data_3.0/1.1.features_pos.txt'
train_paths = []  # list of train input paths
train_rels = []  # list of train relation labels
MAX_PATH_LENGTH = 0
with open(path + filename) as f:
    for line in f:
        line = line.strip()
        line = line.split()
        rel = line[0]
        train_rels.append(rel)
        all_rels.append(rel)
        path = line[1:]
        if len(path) > MAX_PATH_LENGTH:
            MAX_PATH_LENGTH = len(path)
        path = ' '.join(path)
        train_paths.append(path)
        all_paths.append(path)

train_file = open('training_features.pkl', 'rb')
training_map = pickle.load(train_file)
train_file.close()
# print(paths[0])
# print(len(paths[0].split()))
print("longest input path length:", MAX_PATH_LENGTH)

# getting the test feature files
path = 'clean/test_data/'
# just text relations
# filename = 'data_3.0/1.1.test.features.txt'
# POS relations
filename = 'data_3.0/1.1.test.features_pos.txt'
test_paths = []  # list of input paths
test_rels = []  # list of relation labels
with open(path + filename) as f:
    for line in f:
        line = line.strip()
        line = line.split()
        rel = line[0]
        test_rels.append(rel)
        all_rels.append(rel)
        path = line[1:]
        if len(path) > MAX_PATH_LENGTH:
            MAX_PATH_LENGTH = len(path)
        path = ' '.join(path)
        test_paths.append(path)
        all_paths.append(path)

test_file = open('testing_features.pkl', 'rb')
test_map = pickle.load(test_file)
test_file.close()
# print(paths[0])
# print(len(paths[0].split()))
print("longest input path length:", MAX_PATH_LENGTH)

# vectorize text features into 2D integer tensor
MAX_SEQ_LEN = 100
# tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer = Tokenizer(filters='!"#$%&()*+,./:;=?@[\\]^_`{|}~\t\n')  # modified to include --> and <--
tokenizer.fit_on_texts(all_paths)
word_index = tokenizer.word_index
print("Found {} unique tokens".format(len(word_index)))
# print(tokenizer.word_counts)
# print(tokenizer.document_count)
# print(len(rels))
# print(tokenizer.word_index)
# print(tokenizer.word_docs)

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
train_labels = to_categorical(np.asarray(train_labels))
test_labels = le.transform(test_rels)
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
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
# below is just a renaming of test_data and test_labels, for ease of reading
x_test = test_data
y_test = test_labels
print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of x_val: ", x_val.shape)
print("Shape of y_val: ", y_val.shape)
print("Shape of x_test: ", x_test.shape)
print("Shape of y_test: ", y_test.shape)
# print(x_train)
# print(y_train)
# print(x_val)
# print(y_val)
# print(x_test)
# print(y_test)

# prepping the embedding matrix
print("Preparing embedding matrix...")
# getting the pre-trained word embeddings
path = '/home/peace/edu/3/'
filename = 'model.txt'  # NLPL dataset 3
# download from http://vectors.nlpl.eu/repository/ (search for English)
# ID 3, vector size 300, window 5 'English Wikipedia Dump of February 2017'
# vocab size: 296630; Algo: Gensim Continuous Skipgram; Lemma: True
print("Indexing word vectors from", filename)

embeddings_index = {}
with open(path + filename) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        word = word.split('_')[0]  # get just the word, not POS info
        embeddings_index[word] = coefs

print("Found {} word vectors.".format(len(embeddings_index)))

EMBEDDING_DIM = 300
# # num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
num_words = len(word_index) + 1
print(len(word_index)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    # if i > MAX_NUM_WORDS:
    if i > num_words - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
nonzero_elems = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print("word coverage:", nonzero_elems / num_words)
print("embedding_matrix shape:", embedding_matrix.shape)

embedding_layer = Embedding(x_train.shape[1],
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQ_LEN,
                            trainable=True)

# channel 1
inputs1 = Input(shape=[1], dtype='int32')
embed1 = Embedding(output_dim=EMBEDDING_DIM,
                   input_dim=x_train.shape[1],
                   input_length=1,
                   name="Embed1")(inputs1)
vec1 = Reshape([EMBEDDING_DIM])(embed1)

# channel 2
inputs2 = Input(shape=[1], dtype='int32')
embed2 = Embedding(x_train.shape[1],
                   EMBEDDING_DIM,
                   weights=[embedding_matrix],
                   # embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_SEQ_LEN,
                   trainable=True)(inputs2)
# Embedding(1,
#                    EMBEDDING_DIM,
#                    weights=[embedding_matrix],
#                    input_length=num_words,
#                    name="Embed_pretrained",
#                    trainable=True)
vec2 = Reshape([EMBEDDING_DIM])(embed2)

# merge
merged = Concatenate()([vec1, vec2])

# interpretations
dense1 = Dense(512, activation='relu')(merged)
drop1 = Dropout(.2)(dense1)
dense2 = Dense(256, activation='relu')(drop1)
drop2 = Dropout(.2)(dense2)
dense3 = Dense(128, activation='relu')(drop2)
outputs = Dense(1, activation='sigmoid')(dense3)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
# model = Sequential()
# model.add(Embedding(num_words, EMBEDDING_DIM, input_length=MAX_SEQ_LEN))
# model.add(Conv1D(128, 5, activation='relu'))
# model.add(MaxPooling1D(5))
# model.add(Flatten())
# model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc', 'mse', 'cosine'],)
model.summary()
print("model compiled successfully")

print("Training the model...")
start_time = time.time()
history = model.fit([x_train, x_train],
                    y_train,
                    batch_size=128,
                    epochs=10,
                    validation_data=(x_val, y_val),
                    verbose=2)
end_time = np.round(time.time() - start_time, 2)
print("model fitted on x_train, y_train")
print("Training time:", end_time)
print()

print("Evaluating model...")
start_time = time.time()
score, acc, mse_me, cosine = model.evaluate(x_val, y_val, verbose=1)
end_time = np.round(time.time() - start_time, 2)
print("Eval time:", end_time)
print("score:", score)
print("acc:", acc)
print("model mse:", mse_me)
print("cosine:", cosine)
print()

print("Evaluating class predictions on x_test...")
label_names = le.classes_
print(label_names)
start_time = time.time()
y_pred = model.predict(x_test, 128, verbose=2)
end_time = np.round(time.time() - start_time, 2)
print("Predict time:", end_time)
print(classification_report(y_test.argmax(axis=1),
                            y_pred.argmax(axis=1),
                            target_names=label_names))
test_acc = accuracy_score(y_test.argmax(axis=1),
                          y_pred.argmax(axis=1))
print("Accuracy:", test_acc)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("RMSE:", sqrt(mse))
r2 = r2_score(y_test, y_pred)
print("r2:", r2)

