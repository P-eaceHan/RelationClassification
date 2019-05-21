"""
Functions for fast set up of a CNN
@author Peace Han
"""

from math import log
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import Embedding, Flatten, Conv1D, MaxPooling1D


# get the class weights for balanced training
def get_class_weights(labels_dict, le, mu=0.15):
    total = sum(labels_dict.values())
    print(total)
    keys_list = list(labels_dict.keys())
    print(keys_list)
    keys_transformed = le.transform(keys_list)
    # keys_transformed = to_categorical(np.asarray(keys_transformed))
    weights = {}
    for j in range(len(keys_list)):
        label = keys_list[j]
        count = labels_dict[label]
        key_score = log(mu * total / float(count))
        print(label, key_score)
        if key_score > 1.0:
            weights[keys_transformed[j]] = key_score
        elif key_score < 0.0:
            weights[keys_transformed[j]] = 0.1
        else:
            weights[keys_transformed[j]] = 0.5
    return weights


def get_conv_and_pool(x_input, suffix,
                      n_grams=[3, 4, 5],
                      kr=None,
                      feature_maps=100):
    branches = []
    for n in n_grams:
        branch = Conv1D(filters=feature_maps,
                        kernel_size=n,
                        activation='relu',
                        kernel_regularizer=kr,
                        name="conv_{}_{}".format(suffix, n))(x_input)
        branch = Dropout(0.5)(branch)
        branch = MaxPooling1D(pool_size=2,
                              strides=None,
                              padding='valid',
                              name="pool_{}_{}".format(suffix, n))(branch)
        branch = Flatten(name="flat_{}_{}".format(suffix, n))(branch)
        branches.append(branch)
    return branches


def get_conv_and_pool2(x_input, suffix, n_grams=[3, 4, 5], feature_maps=100):
    branches = []
    for n in n_grams:
        branch = Conv1D(filters=feature_maps,
                        kernel_size=n,
                        activation='relu',
                        name="conv_{}_{}".format(suffix, n))(x_input)
        branch = MaxPooling1D(pool_size=2,
                              strides=None,
                              padding='valid',
                              name="pool_{}_{}".format(suffix, n))(branch)
        branch = Conv1D(filters=feature_maps,
                        kernel_size=n,
                        activation='relu',
                        name="conv2_{}_{}".format(suffix, n))(x_input)
        branch = MaxPooling1D(pool_size=2,
                              strides=None,
                              padding='valid',
                              name="pool2_{}_{}".format(suffix, n))(branch)
        branch = Flatten(name="flat_{}_{}".format(suffix, n))(branch)
        branches.append(branch)
    return branches

