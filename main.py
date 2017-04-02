import xlrd
import numpy as np
from scipy.spatial import distance

# coding=utf-8
# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 :

START_COLUMN = 1
NUMBER_OF_FIRST_MEASUREMENT = 16
list_of_files = ['belkozin_21_07_14_17_48.XLS', 'vodka_proba_1_21_05_14_18_35.XLS', 'voda_proba_1_20_05_14_19_33.XLS',
                 'beef_50_23_05_14_14_42.XLS', 'cheeken_50_23_05_14_14_12.XLS', 'aceton_14_05_14_13_20.XLS',
                 'izopropanol_14_05_14_16_22.XLS', 'ethylacetate_16_05_14_15_49.XLS', 'uksus_2_09_14_14_13.XLS',
                 'voda_distil_17_09_14_15_20.XLS', 'etanol_3_07_14_17_25.XLS', 'aceton_1_07_14_15_45.XLS',
                 'etanol_4_07_14_16_40.XLS', 'suharik_29_06_09_17_33.XLS', 'suharik_29_06_09_17_36.XLS']


def parsing(path_to_file):
    workbook = xlrd.open_workbook(path_to_file)
    sheet = workbook.sheet_by_index(0)
    count = 0
    values = []
    arr = []
    for col in range(START_COLUMN, sheet.ncols):
        for elem in workbook.sheet_by_index(0).col(col):
            count += 1
            if count > 10:
                values.append(elem.value)
        if col == START_COLUMN:
            arr = np.array(values)
        else:
            arr = np.vstack((arr, values))
        count = 0
        values = []
    return np.transpose(arr)


def get_information_from_measurment():
    count = 0
    result = np.array([])
    for path_to_file in list_of_files:
        vector = []
        print path_to_file + ' - ' + str(count)
        arr = parsing('./resource/' + path_to_file)

        sensors = [0, 1, 2, 3, 4, 5, 6, 7]
        for col in sensors:
            first_max = np.array(arr[0:NUMBER_OF_FIRST_MEASUREMENT, col]).max()
            last_mean = np.array(arr[NUMBER_OF_FIRST_MEASUREMENT:arr.shape[0], col]).mean()
            # ratio = abs(first_max / last_mean)
            vector.append(first_max)
            vector.append(last_mean)
            # vector.append(ratio)
        if result.shape[0] == 0:
            result = np.append(result, vector)
        else:
            result = np.vstack((result, vector))
        count += 1
    return result.transpose()


# method computes closeness of dest vector with src_1 and src_2 vectors based on euclidean measurement
def closeness(src_1, src_2, dest):
    dist_1 = distance.euclidean(src_1, dest)
    print str(dist_1) + ' is closeness between first measurement and destination vector'
    dist_2 = distance.euclidean(src_2, dest)
    print str(dist_2) + ' is closeness between second measurement and destination vector'

    if dist_1 <= dist_2:
        print str(dist_1) + ' is better'
        return src_1
    else:
        print str(dist_2) + ' is better'
        return src_2


# compute distance between all measurement
def closeness_all(index):
    for i in range(0, len(list_of_files)):
        print str(distance.euclidean(result[:, i], result[:, index])) + ' - ' + list_of_files[i]


result = get_information_from_measurment()
print '\n'

closeness(result[:, 0], result[:, 1], result[:, 12])

print '\n'
closeness_all(10)
