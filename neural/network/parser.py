import xlrd
import numpy as np
from os import listdir
from os.path import isfile, join

PATH_TO_DIR = './resource'
SPLIT_SYMBOL = '/'
DATE = ['29.04.16', '03.05.16']
SUBSTANCE = ['acetaldehyde', 'acetone', 'butanol-1', 'butanol-2', 'ethanol', 'MEK',
             'propanol-1', 'propanol-2', 'toluene']
NAME = ['measurement-1.xlsx', 'measurement-2.xlsx', 'measurement-3.xlsx']


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


def get_standard_measurement(date, substance, name):
    START_NAME = 29
    file_name = PATH_TO_DIR + '/standard/' + DATE[date] + SPLIT_SYMBOL + SUBSTANCE[substance] \
                + SPLIT_SYMBOL + NAME[name]
    print(parser_stand(file_name))
    print '\nFILE NAME: ' + file_name[START_NAME:]


def get_other_measurement(file_name):
    path_to_file = PATH_TO_DIR + SPLIT_SYMBOL + file_name
    print(parser_other(path_to_file))
    print '\nFILE NAME: ' + path_to_file


def get_list_of_files_in_dir(path_to_dir, isprint):
    only_files = [f for f in listdir(path_to_dir) if isfile(join(path_to_dir, f))]

    if isprint:
        index = 0
        for f in only_files:
            print f + ' - ' + str(index)
            index += 1

    return only_files


# file_list = get_list_of_files_in_dir(PATH_TO_DIR, False)

# get_standard_measurement(1, 3, 1)

# get_other_measurement(file_list[0])
