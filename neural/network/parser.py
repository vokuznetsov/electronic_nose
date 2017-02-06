import xlrd
import numpy as np
from os import listdir
from os.path import isfile, join

PATH_TO_DIR = './resource'
SPLIT_SYMBOL = '/'

# standard resource folder contains only measurement with 6 sensors
DATE = ['29.04.16', '03.05.16']
SUBSTANCE = ['acetaldehyde', 'acetone', 'butanol-1', 'butanol-2', 'ethanol', 'MEK',
             'propanol-1', 'propanol-2', 'toluene']
MEASUREMENT = ['measurement-1.xlsx', 'measurement-2.xlsx', 'measurement-3.xlsx']


# parse standard measurement
def parser_stand(path_to_file):
    START_COLUMN = 6
    workbook = xlrd.open_workbook(path_to_file)
    sheet = workbook.sheet_by_index(0)
    values = []
    arr = []
    for col in range(START_COLUMN, sheet.ncols):
        for elem in workbook.sheet_by_index(0).col(col):
            values.append(elem.value)
        if col == START_COLUMN:
            arr = np.array(values)
        else:
            arr = np.vstack((arr, values))
        values = []
    return np.transpose(arr)


# parse not standard measurement
def parser_other(path_to_file):
    START_COLUMN = 1
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


def get_standard_measurement(date, substance, measurement):
    START_NAME = 29
    file_name = PATH_TO_DIR + '/standard/' + DATE[date] + SPLIT_SYMBOL + SUBSTANCE[substance] \
                + SPLIT_SYMBOL + MEASUREMENT[measurement]
    standard = parser_stand(file_name)
    # print(standard)
    print '\nFILE NAME: ' + file_name[START_NAME:]
    return standard


def get_other_measurement(file_name):
    path_to_file = PATH_TO_DIR + SPLIT_SYMBOL + file_name
    other = parser_other(path_to_file)
    # print('\n' + str(other))
    print '\nFILE NAME: ' + path_to_file
    return other


# reformat from shape (m x n) to (1 x (m*n)), where first m line is 1 column, second m line is 2 column and etc.
def reformat_measurement(measurement):
    result = np.array(measurement[:, 0])

    for i in range(1, measurement.shape[1]):
        column = measurement[:, i]
        result = np.concatenate((result, column))
    return result


def get_all_standard_data():
    data = []

    for d in range(0, len(DATE)):
        print DATE[d]
        for s in range(0, len(SUBSTANCE)):
            for m in range(0, len(MEASUREMENT)):
                st = get_standard_measurement(d, s, m)
                data.append(reformat_measurement(st))
    return data


def get_list_of_files_in_dir(path_to_dir, isprint):
    only_files = [f for f in listdir(path_to_dir) if isfile(join(path_to_dir, f))]

    if isprint:
        index = 0
        for f in only_files:
            print f + ' - ' + str(index)
            index += 1

    return only_files

# standard = get_standard_measurement(1, 3, 1)
# file_list = get_list_of_files_in_dir(PATH_TO_DIR, False)
# other = get_other_measurement(file_list[0])

# reformat_st = reformat_measurement(standard)
# reformat_oth = reformat_measurement(other)
#
# print 'Standard shape: ' + str(standard.shape) + '\n'
# print 'Reformat shape: ' + str(reformat_st.shape) + '\n'
#
# print 'Other shape: ' + str(other.shape) + '\n'
# print 'Reformat shape: ' + str(reformat_oth.shape) + '\n'
