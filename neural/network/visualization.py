import numpy as np
import parser

standard_data = parser.get_all_standard_data()
non_stadard_data = parser.get_all_non_standard_data()


# print len(standard_data)
# print len(non_stadard_data)


def get_train_data():
    data = np.dstack((standard_data[0], non_stadard_data[0], standard_data[1]))
    return np.array(data).reshape(1, 120, 10, 3)


def get_train_labels():
    labels = np.array(1).reshape(1,1)
    return labels


train_data = get_train_data()
labels = get_train_labels()

print train_data
print labels

# array = parser.get_standard_measurement(1, 5, 2)

# file_list = parser.get_list_of_files_in_dir(parser.PATH_TO_DIR, False)
# other = parser.get_other_measurement(file_list[3])

# parser.create_image_from_array(array)
