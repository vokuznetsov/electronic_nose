import numpy as np
import parser

FILE_SIZE = 750
DATA_NAME = 'data'
LABELS_NAME = 'label'

standard_data = parser.get_all_standard_data()
non_standard_data = parser.get_all_non_standard_data()

def get_train_data():
    data1 = np.dstack((standard_data[0], non_standard_data[0], standard_data[1]))
    d = np.array(data1).reshape(1, 120, 10, 3)

    data2 = np.dstack((standard_data[1], non_standard_data[0], standard_data[1])).reshape(1, 120, 10, 3)
    data3 = np.dstack((standard_data[9], non_standard_data[3], standard_data[18])).reshape(1, 120, 10, 3)
    data4 = np.dstack((non_standard_data[3], standard_data[11], standard_data[6])).reshape(1, 120, 10, 3)

    d = np.append(d, data2, axis=0)
    d = np.append(d, data3, axis=0)
    d = np.append(d, data4, axis=0)
    return d


def get_train_labels():
    labels = np.array((1, 1, 1, 0)).reshape(4, 1)
    return labels


def get_test_data():
    data1 = np.dstack((non_standard_data[10], standard_data[10], standard_data[3])).reshape(1, 120, 10, 3)
    data2 = np.dstack((non_standard_data[7], standard_data[1], standard_data[14])).reshape(1, 120, 10, 3)
    data3 = np.dstack((non_standard_data[6], standard_data[12], standard_data[6])).reshape(1, 120, 10, 3)
    data4 = np.dstack((non_standard_data[3], standard_data[8], standard_data[19])).reshape(1, 120, 10, 3)
    data5 = np.dstack((standard_data[20], non_standard_data[1], standard_data[15])).reshape(1, 120, 10, 3)

    d = np.array(data1).reshape(1, 120, 10, 3)
    d = np.append(d, data2, axis=0)
    d = np.append(d, data3, axis=0)
    d = np.append(d, data4, axis=0)
    d = np.append(d, data5, axis=0)
    return d


def get_test_labels():
    labels = np.array((0, 0, 0, 0, 1)).reshape(5, 1)
    return labels


def compressed_data(filename, data, label):
    np.savez_compressed('resource/data.archive/' + filename, data=data, label=label)


def archive_data():
    standard_data = parser.get_all_standard_data()
    non_stadard_data = parser.get_all_non_standard_data()
    SIZE_OF_DATA = len(standard_data) * len(non_stadard_data) * len(non_stadard_data)

    data, label, values = collect_data(np.array((0, -1), dtype=np.int16), standard_data, non_stadard_data)
    compressed_data("data_" + str(values[1]), data, label)
    while values[1] <= SIZE_OF_DATA:
        data, label, values = collect_data(values, standard_data, non_stadard_data)
        compressed_data("data_" + str(values[1]), data=data, label=label)


def read_data(filename):
    return np.load('resource/data.archive/' + filename + '.npz')


def collect_data(value, standard_data, non_standard_data):
    data = []
    labels = []
    count = value[1]
    for i in range(value[0], len(standard_data)):
        s = standard_data[i]
        for j in range(0, len(non_standard_data)):
            ns = non_standard_data[j]
            for k in range(0, len(standard_data)):
                sub = standard_data[k]
                count += 1

                if count % 500 == 0:
                    print 'Count is ' + str(count) + ' from ' + \
                          str(len(standard_data) * len(non_standard_data) * len(non_standard_data))
                if count != 0 and count % FILE_SIZE == 0:
                    return data, labels, np.array((i, count), dtype=np.int16)



                false = np.dstack((ns, s, sub)).reshape(1, 120, 10, 3)
                if len(data) == 0:
                    data = np.array(true).reshape(1, 120, 10, 3)
                    labels = np.array(1)
                else:
                    data = np.append(data, true, axis=0)
                    labels = np.append(labels, 1)
                data = np.append(data, false, axis=0)
                labels = np.append(labels, 0)
    return data, labels, np.array((len(standard_data), count), dtype=np.int16)


# archive_data()
#
# arr = read_data("data_22500")
# print arr[DATA_NAME].shape
# print arr[LABELS_NAME].shape


# array = parser.get_standard_measurement(1, 5, 2)

# file_list = parser.get_list_of_files_in_dir(parser.PATH_TO_DIR, False)
# other = parser.get_other_measurement(file_list[3])

# parser.create_image_from_array(array)
