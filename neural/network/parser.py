# coding=utf-8
import xlrd
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
from PIL import Image

MAX_VALUE = -32000
PATH_TO_DIR = './resource'
STANDARD_DATA = '/standard/'
NON_STANDARD_DATA = '/non_standard/'
ALCOHOL_DATA = "/alcohol/"
SPLIT_SYMBOL = '/'

# standard resource folder contains only measurement with 6 sensors
SENSORS_STANDARD = ["Прополис", "МУНТ", "ПЭГ-2000", "ТОФО", "Тритон", "ДЦГ-18К6", "ПЭГС", "ПФЭ", "ПЭГСб", "ПЭГФ"]

DATE = ['29.04.16', '03.05.16-4', '03.05.16-6', '04.05.16']
SUBSTANCE = ['acetaldehyde', 'acetone', 'butanol-1', 'butanol-2', 'ethanol', 'MEK',
             'propanol-1', 'propanol-2', 'toluene']
MEASUREMENT = ['measurement-1.xlsx', 'measurement-2.xlsx', 'measurement-3.xlsx', 'measurement-4.xlsx']

ALCOHOL_DATE = ['13.05.16', '14.05.16', '16.05.16']
ALCOHOL_SUBSTANCE = 10


def is_max(elem):
    global MAX_VALUE
    if MAX_VALUE < elem:
        MAX_VALUE = elem


# parse standard measurement
def parser(path_to_file, is_standard):
    START_COLUMN_STANDARD = 5
    START_COLUMN_OTHER = 1
    START_ROW_OTHER = 10
    try:
        workbook = xlrd.open_workbook(path_to_file)
        sensors = get_list_of_sensors(path_to_file, is_standard)
    except IOError:
        return
    values = []
    arr = []
    count = 0

    for s in SENSORS_STANDARD:
        if s in sensors:
            index = sensors.index(s)

            if is_standard:
                for elem in workbook.sheet_by_index(0).col(START_COLUMN_STANDARD + index):
                    is_max(elem.value)
                    values.append(elem.value)
            else:
                for elem in workbook.sheet_by_index(0).col(START_COLUMN_OTHER + index):
                    count += 1
                    if count > START_ROW_OTHER:
                        is_max(elem.value)
                        if len(values) == 120:
                            break
                        values.append(elem.value)

            last_value = values[-1]
            # Neural network take a matrix with 120 rows so it is necessary to add additional rows with 0 value
            # in order to get a vector with size 120x1.
            for i in range(len(values), 120):
                values.append(last_value)
        else:
            # so we should have a matrix with size 120x10, we should add a zeros vector with size (120x1)
            # zeros vectors because the sensors is not contained in SENSORS_STANDARD
            values = np.zeros(120)

        if len(arr) == 0:
            arr = np.array(values)
        else:
            arr = np.vstack((arr, values))
        values = []
        count = 0

    return np.transpose(arr)


def get_standard_measurement(date, substance, measurement):
    START_NAME = 29
    file_name = PATH_TO_DIR + STANDARD_DATA + DATE[date] + SPLIT_SYMBOL + SUBSTANCE[substance] \
                + SPLIT_SYMBOL + MEASUREMENT[measurement]
    standard = parser(file_name, is_standard=True)
    # create_image_from_array(standard, True, DATE[date], SUBSTANCE[substance], MEASUREMENT[measurement][-6])
    # print(standard)
    print '\nFILE NAME: ' + file_name[START_NAME:]
    return standard


def get_other_measurement(file_name):
    path_to_file = PATH_TO_DIR + NON_STANDARD_DATA + 'data/' + file_name
    other = parser(path_to_file, is_standard=False)
    # create_image_from_array(array=other, is_standard=False, filename=file_name)
    # print('\n' + str(other))
    print '\nFILE NAME: ' + path_to_file
    return other


def get_alcohol_measurement(date, substance, measurement):
    START_NAME = 0
    file_name = PATH_TO_DIR + ALCOHOL_DATA + ALCOHOL_DATE[date] + SPLIT_SYMBOL + str(substance) \
                + SPLIT_SYMBOL + MEASUREMENT[measurement]

    alcohol = parser(file_name, is_standard=True)
    print '\nFILE NAME: ' + file_name[START_NAME:]
    return alcohol


def get_list_of_sensors(path_to_file, is_standard):
    values = []
    workbook = xlrd.open_workbook(path_to_file)

    if is_standard:
        SENSORS_COLUMN = 3
        for elem in workbook.sheet_by_index(0).col(SENSORS_COLUMN):
            s = elem.value.encode('utf-8').strip()
            if s == "":
                break
            values.append(s)
    else:
        SENSORS_COLUMN = 1
        SENSORS_ROW = 6
        count = 0
        for elem in workbook.sheet_by_index(0).row(SENSORS_ROW):
            count += 1
            if count > SENSORS_COLUMN:
                s = elem.value.split(" ")
                sensor = s[0].encode('utf-8').strip()
                if sensor == "":
                    break
                values.append(sensor)

    return values


# reformat from shape (m x n) to (1 x (m*n)), where first m line is 1 column, second m line is 2 column and etc.
def reformat_measurement(measurement):
    result = np.array(measurement[:, 0])

    for i in range(1, measurement.shape[1]):
        column = measurement[:, i]
        result = np.concatenate((result, column))
    return result


def get_list_of_files_in_dir(path_to_dir, isprint=False):
    only_files = [f for f in listdir(path_to_dir) if isfile(join(path_to_dir, f))]

    if isprint:
        index = 0
        for f in only_files:
            print f + ' - ' + str(index)
            index += 1

    return only_files


def get_all_standard_data(is_reformat=False):
    data = []

    for d in range(0, len(DATE)):
        print DATE[d]
        for s in range(0, len(SUBSTANCE)):
            for m in range(0, len(MEASUREMENT)):
                # 03.05.16-4 contains only 2 measurements for each substance,
                # so we need to skip 3-rd measurement
                # if DATE[d] == DATE[1] and m == 2:
                #     continue
                st = get_standard_measurement(d, s, m)
                if st is None:
                    continue
                elif is_reformat:
                    data.append(reformat_measurement(st))
                else:
                    data.append(st)
    return data


def get_all_non_standard_data(is_reformat=False):
    NON_STANDARD_DIR = PATH_TO_DIR + NON_STANDARD_DATA + 'data/'
    list_of_files = get_list_of_files_in_dir(NON_STANDARD_DIR)

    data = collect_non_standard_data(list_of_files, is_reformat)
    return data


def get_all_alcohol(is_reformat=False):
    data = []

    for d in range(0, len(ALCOHOL_DATE)):
        print ALCOHOL_DATE[d]
        for s in range(1, ALCOHOL_SUBSTANCE + 1):
            for m in range(0, len(MEASUREMENT)):
                alc = get_alcohol_measurement(d, s, m)

                if alc is None:
                    continue
                elif is_reformat:
                    data.append(reformat_measurement(alc))
                else:
                    data.append(alc)
    return data


def get_all_non_standard_data_splitted_by_alcohol():
    NON_STANDARD_ALC_DIR = PATH_TO_DIR + NON_STANDARD_DATA + 'data_test/non_stand_alc'
    NON_STANDARD_NON_ALC_DIR = PATH_TO_DIR + NON_STANDARD_DATA + 'data_test/non_stand_non_alc'

    list_of_files_alc = get_list_of_files_in_dir(NON_STANDARD_ALC_DIR)
    list_of_files_non_alc = get_list_of_files_in_dir(NON_STANDARD_NON_ALC_DIR)

    non_stand_alc = collect_non_standard_data(list_of_files_alc)
    non_stand_non_alc = collect_non_standard_data(list_of_files_non_alc)

    return non_stand_alc, non_stand_non_alc


def collect_non_standard_data(list_of_files, is_reformat=False):
    data = []
    for f in list_of_files:
        non_st = get_other_measurement(f)
        if non_st is None:
            continue
        elif is_reformat:
            data.append(reformat_measurement(non_st))
        else:
            data.append(non_st)
    return data


def create_image_from_array(array, is_standard=False, date=None, substance=None, measurement=None, filename=None):
    formatted = (array * 255 / np.max(array)).astype('uint8')
    img = Image.fromarray(formatted, 'L')

    if is_standard:
        path = PATH_TO_DIR + SPLIT_SYMBOL + STANDARD_DATA + 'images' + SPLIT_SYMBOL

        if not exists(path + date):
            makedirs(path + date)
        if not exists(path + SPLIT_SYMBOL + date + SPLIT_SYMBOL + substance):
            makedirs(path + SPLIT_SYMBOL + date + SPLIT_SYMBOL + substance)
        if not exists(path + 'all'):
            makedirs(path + 'all')

        img.save(str(path + SPLIT_SYMBOL + date + SPLIT_SYMBOL + substance + SPLIT_SYMBOL + measurement) + '.png')
        img.save(str(path + SPLIT_SYMBOL + 'all' + SPLIT_SYMBOL + substance + '-' + date + '-' + measurement)
                 + '.png')
    else:
        path = PATH_TO_DIR + SPLIT_SYMBOL + NON_STANDARD_DATA + 'images' + SPLIT_SYMBOL
        if not exists(path + 'all'):
            makedirs(path + 'all')
        img.save(str(path + SPLIT_SYMBOL + 'all' + SPLIT_SYMBOL + filename) + '.png')

# get_all_non_standard_data()
# get_all_non_standard_data_splitted_by_alcohol()

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
