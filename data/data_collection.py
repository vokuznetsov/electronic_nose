import numpy as np
import parser
import random
import tensorflow as tf

STANDARD_PATH = "normalize_data/standard"
NON_STANDARD_PATH = "normalize_data/non_standard"
MODEL_GOOD_PATH = "normalize_data/model_good"
MODEL_BAD_PATH = "normalize_data/model_bad"

# non_standard_alc_data = np.load('resource/data.archive/' + NON_STANDARD_PATH +
#                                 '/non_stand_alc' + '.npz')["data"]
# non_standard_non_alc_data = np.load('resource/data.archive/' + NON_STANDARD_PATH +
#                                     '/non_stand_non_alc' + '.npz')["data"]

standard_data = np.load('resource/' + STANDARD_PATH + '.npz')["data"]
non_standard_data = np.load('resource/' + NON_STANDARD_PATH + '.npz')["data"]
model_good_data = np.load('resource/' + MODEL_GOOD_PATH + '.npz')["data"]
model_bad_data = np.load('resource/' + MODEL_BAD_PATH + '.npz')["data"]

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
    standard_data = parser.get_all_st_non_st_data(is_standard=True)
    non_stadard_data = parser.get_all_st_non_st_data(is_standard=False)
    model_good_data = parser.get_model_data(is_good=True)
    model_bad_data = parser.get_model_data(is_good=False)

    standard_data = normalization(standard_data)
    non_stadard_data = normalization(non_stadard_data)
    model_good_data = normalization(model_good_data)
    model_bad_data = normalization(model_bad_data)

    np.savez_compressed('resource/' + STANDARD_PATH, data=standard_data)
    np.savez_compressed('resource/' + NON_STANDARD_PATH, data=non_stadard_data)
    np.savez_compressed('resource/' + MODEL_GOOD_PATH, data=model_good_data)
    np.savez_compressed('resource/' + MODEL_BAD_PATH, data=model_bad_data)


def get_data(batch_size, start_elem, end_elem, standard_data, st_start, st_end,
             non_standard_data, non_st_start, non_st_end, model_good_data, model_bad_data):
    data = []
    labels = []
    for i in range(0, batch_size):
        st_index = random.randint(st_start, st_end)
        non_st_index = random.randint(non_st_start, non_st_end)
        label = np.array(random.randint(0, 1)).reshape(1, 1)

        if label == 0:
            for good in model_good_data:
                d = np.dstack((non_standard_data[non_st_index], standard_data[st_index], good)).reshape(1, 120, 10, 3)
                data, labels = add_to_data(start_elem, end_elem, d, data, label, labels)

            for bad in model_bad_data:
                d = np.dstack((standard_data[st_index], non_standard_data[non_st_index], bad)).reshape(1, 120, 10, 3)
                data, labels = add_to_data(start_elem, end_elem, d, data, label, labels)
        else:
            for good in model_good_data:
                d = np.dstack((standard_data[st_index], non_standard_data[non_st_index], good)).reshape(1, 120, 10, 3)
                data, labels = add_to_data(start_elem, end_elem, d, data, label, labels)

            for bad in model_bad_data:
                d = np.dstack((non_standard_data[non_st_index], standard_data[st_index], bad)).reshape(1, 120, 10, 3)
                data, labels = add_to_data(start_elem, end_elem, d, data, label, labels)

    return data, labels


def get_train_data(batch_size, start_elem, end_elem):
    st_end = round(len(standard_data) * PERCENT_OF_TRAINING_DATA)
    non_st_end = round(len(non_standard_data) * PERCENT_OF_TRAINING_DATA)

    data, labels = get_data(batch_size, start_elem, end_elem, standard_data, 0, st_end,
                            non_standard_data, 0, non_st_end, model_good_data, model_bad_data)
    return data, labels


def get_test_data(batch_size, start_elem, end_elem):
    st_start = round(len(standard_data) * PERCENT_OF_TRAINING_DATA)
    non_st_start = round(len(non_standard_data) * PERCENT_OF_TRAINING_DATA)

    st_end = len(standard_data) - 1
    non_st_end = len(non_standard_data) - 1

    data, labels = get_data(batch_size, start_elem, end_elem, standard_data, st_start, st_end, non_standard_data,
                            non_st_start, non_st_end, model_good_data, model_bad_data)
    return data, labels


def add_to_data(start_elem, end_elem, d, data, label, labels):
    d = d[:, start_elem:end_elem, :, :]
    if len(data) == 0:
        data = np.array(d).reshape(1, (end_elem - start_elem), 10, 3)
        labels = np.array(label).reshape(1, 1)
    else:
        data = np.append(data, d, axis=0)
        labels = np.append(labels, label, axis=0)
    return data, labels

# archive_parser_data()
# archive_non_stand_data_splitted_by_alc()
