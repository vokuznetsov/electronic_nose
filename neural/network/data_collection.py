import numpy as np
import parser
import random
import tensorflow as tf

STANDARD_PATH = "normalization/standard"
NON_STANDARD_PATH = "normalization/non_standard"
ALCOHOL_PATH = "normalization/alcohol"

# non_standard_alc_data = np.load('resource/data.archive/' + NON_STANDARD_PATH +
#                                 '/non_stand_alc' + '.npz')["data"]
# non_standard_non_alc_data = np.load('resource/data.archive/' + NON_STANDARD_PATH +
#                                     '/non_stand_non_alc' + '.npz')["data"]

standard_data = np.load('resource/data.archive/' + STANDARD_PATH + '.npz')["data"]
non_standard_data = np.load('resource/data.archive/' + NON_STANDARD_PATH + '.npz')["data"]
alcohol_data = np.load('resource/data.archive/' + ALCOHOL_PATH + '.npz')["data"]

PERCENT_OF_TRAINING_DATA = 0.7


def normalization(data):
    init = tf.initialize_all_variables()
    x = np.array(data).reshape(-1, 120, 10)
    x = tf.reshape(x, [-1, 120, 10])
    normed = tf.nn.l2_normalize(x, dim=1)
    sess = tf.Session()
    sess.run(init)
    return sess.run(normed)


def archive_parser_data():
    standard_data = parser.get_all_standard_data()
    non_stadard_data = parser.get_all_non_standard_data()
    alcohol_data = parser.get_all_alcohol()

    standard_data = normalization(standard_data)
    non_stadard_data = normalization(non_stadard_data)
    alcohol_data = normalization(alcohol_data)

    np.savez_compressed('resource/data.archive/' + STANDARD_PATH, data=standard_data)
    np.savez_compressed('resource/data.archive/' + NON_STANDARD_PATH, data=non_stadard_data)
    np.savez_compressed('resource/data.archive/' + ALCOHOL_PATH, data=alcohol_data)


def archive_non_stand_data_splitted_by_alc():
    non_stand_alc, non_stand_non_alc = parser.get_all_non_standard_data_splitted_by_alcohol()

    non_stand_alc = normalization(non_stand_alc)
    non_stand_non_alc = normalization(non_stand_non_alc)

    np.savez_compressed('resource/data.archive/' + NON_STANDARD_PATH + '/non_stand_alc',
                        data=non_stand_alc)
    np.savez_compressed('resource/data.archive/' + NON_STANDARD_PATH + '/non_stand_non_alc',
                        data=non_stand_non_alc)


def get_data(batch_size, standard_data, st_start, st_end, non_standard_data, non_st_start, non_st_end,
             alcohol_data, alc_start, alc_end):
    data = []
    labels = []
    for i in range(0, batch_size):
        st_index = random.randint(st_start, st_end)
        non_st_index = random.randint(non_st_start, non_st_end)
        alc_index = random.randint(alc_start, alc_end)
        label = np.array(random.randint(0, 1)).reshape(1, 1)

        if label == 0:
            d = np.dstack((non_standard_data[non_st_index], standard_data[st_index], alcohol_data[alc_index])).reshape(
                1, 120, 10, 3)
        else:
            d = np.dstack((standard_data[st_index], non_standard_data[non_st_index], alcohol_data[alc_index])).reshape(
                1, 120, 10, 3)

        if len(data) == 0:
            data = np.array(d).reshape(1, 120, 10, 3)
            labels = np.array(label).reshape(1, 1)
        else:
            data = np.append(data, d, axis=0)
            labels = np.append(labels, label, axis=0)
    return data, labels


def get_train_data(batch_size):
    st_end = round(len(standard_data) * PERCENT_OF_TRAINING_DATA)
    non_st_end = round(len(non_standard_data) * PERCENT_OF_TRAINING_DATA)
    alc_end = round(len(alcohol_data) * PERCENT_OF_TRAINING_DATA)

    data, labels = get_data(batch_size, standard_data, st_start, st_end, non_standard_data, non_st_start, non_st_end,
                            alcohol_data, alc_start, alc_end)
    return data, labels


def get_test_data(batch_size):
    st_start = round(len(standard_data) * PERCENT_OF_TRAINING_DATA)
    non_st_start = round(len(non_standard_data) * PERCENT_OF_TRAINING_DATA)
    alc_start = round(len(alcohol_data) * PERCENT_OF_TRAINING_DATA)

    st_end = len(standard_data) - 1
    non_st_end = len(non_standard_data) - 1
    alc_end = len(alcohol_data) - 1

    data, labels = get_data(batch_size, standard_data, st_start, st_end, non_standard_data, non_st_start, non_st_end,
                            alcohol_data, alc_start, alc_end)
    return data, labels


def get_test_non_stand_splitted_by_alc_data(batch_size):
    alc_end_size = len(non_standard_alc_data) - 1
    non_alc_end_size = len(non_standard_non_alc_data) - 1

    data, labels = get_data(batch_size, non_standard_alc_data, 0, alc_end_size, non_standard_non_alc_data, 0,
                            non_alc_end_size,
                            alcohol_data, 0, len(alcohol_data) - 1)

    return data, labels


def get_test_data_for_all_alc(batch_size):
    data = []
    labels = []

    st_start = round(len(standard_data) * PERCENT_OF_TRAINING_DATA)
    non_st_start = round(len(non_standard_data) * PERCENT_OF_TRAINING_DATA)
    st_end = len(standard_data) - 1
    non_st_end = len(non_standard_data) - 1

    for i in range(0, batch_size):
        st_index = random.randint(st_start, st_end)
        non_st_index = random.randint(non_st_start, non_st_end)

        for j in range(0, len(alcohol_data) - 1):
            label = np.array(random.randint(0, 1)).reshape(1, 1)
            if label == 0:
                d = np.dstack(
                    (non_standard_data[non_st_index], standard_data[st_index], alcohol_data[j])) \
                    .reshape(1, 120, 10, 3)
            else:
                d = np.dstack(
                    (standard_data[st_index], non_standard_data[non_st_index], alcohol_data[j])) \
                    .reshape(1, 120, 10, 3)

            if len(data) == 0:
                data = np.array(d).reshape(1, 120, 10, 3)
                labels = np.array(label).reshape(1, 1)
            else:
                data = np.append(data, d, axis=0)
                labels = np.append(labels, label, axis=0)
    return data, labels


    # archive_parser_data()
    # archive_non_stand_data_splitted_by_alc()
    #
    # print len(standard_data)
    # print len(non_standard_data)
    # print len(alcohol_data)
    #
    #
    # print round(len(standard_data) * PERCENT_OF_TRAINING_DATA)
    # print round(len(non_standard_data) * PERCENT_OF_TRAINING_DATA)
    # print round(len(alcohol_data) * PERCENT_OF_TRAINING_DATA)
