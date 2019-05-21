"""
Code to generate the word embedding matrix from pretrained word embeddings.
Instructions to obtain embeddings in comments below.
Run this before running classify_rels.py
@author Peace Han
"""
import pickle
import time
import numpy as np
# from gensim.models import KeyedVectors

# prepping the embedding matrix
print("Preparing embedding matrix...")
# getting the pre-trained word embeddings
# path = '/home/peace/edu/3/'
path = '../../'
# download from http://vectors.nlpl.eu/repository/ (search for English)
# ID 3, vector size 300, window 5 'English Wikipedia Dump of February 2017'
# vocab size: 296630; Algo: Gensim Continuous Skipgram; Lemma: True
filename = '3/model.txt'
print("Indexing word vectors from", filename)
start_time = time.time()
embeddings_index = {}  # just word embeddings
pos_embeddings_index = {}  # just POS embeddings
# combined_embeddings_index = {}  # both word and POS embeddings
with open(path + filename) as f:
    next(f)
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        words = word.split('_')
        # print(words)
        word = words[0]  # get just the word, not POS info
        pos = words[1]   # get just the POS
        embeddings_index[word] = coefs
        # combined_embeddings_index[word] = coefs
        if pos not in pos_embeddings_index.keys():
            pos_embeddings_index[pos] = (np.full_like(coefs, 0), 0)
        # if pos not in combined_embeddings_index.keys():
        #     combined_embeddings_index[pos] = (np.full_like(coefs, 0), 0)
        pos_orig_coefs = pos_embeddings_index[pos]
        # combo_orig_coeffs = combined_embeddings_index[pos]
        pos_embeddings_index[pos] = (pos_orig_coefs[0] + coefs, pos_orig_coefs[1])
        # combined_embeddings_index = (combo_orig_coeffs[0] + coefs, combo_orig_coeffs[1])

embeddings_file = open('embeddings_index.pkl', 'wb')
pickle.dump(embeddings_index, embeddings_file)
embeddings_file.close()
# print(pos_embeddings_index)
pos_embeddings_file = open('pos_embeddings_index.pkl', 'wb')
pickle.dump(pos_embeddings_index, pos_embeddings_file)
pos_embeddings_file.close()
# comb_embeddings_file = open('combo_index.pkl', 'wb')
# pickle.dump(combined_embeddings_index, comb_embeddings_file)
# comb_embeddings_file.close()
end_time = np.round(time.time() - start_time, 2)
print("Found {} word vectors in embeddings index.".format(len(embeddings_index)))
print("Found {} word vectors in pos embeddings index.".format(len(pos_embeddings_index)))
# print("Found {} word vectors in combo embeddings index.".format(len(combined_embeddings_index)))
print("Time to fetch and save word embeddings: {}s".format(end_time))

# # Word2vec vectors
# filename = 'GoogleNews-vectors-negative300.bin'  # word2vec vectors
# print("Loading word vectors from", filename)
# start_time = time.time()
# word2vec = KeyedVectors.load_word2vec_format(path + filename, binary=True)
# w2v_file = open("w2v_index.pkl", 'wb')
# print("Dumping to pkl file...")
# pickle.dump(word2vec, w2v_file)
# w2v_file.close()
#
# end_time = np.round(time.time() - start_time, 2)
# print("Found {} word vectors in word2vec".format(len(word2vec)))
# print("Time to fetch and save word embeddings: {}s".format(end_time))
#
f.close()

