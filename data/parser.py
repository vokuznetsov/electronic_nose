# coding=utf-8
import xlrd
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
from PIL import Image

PATH_TO_DIR = './resource'
STANDARD_DATA = '/data/standard/'
NON_STANDARD_DATA = '/data/non_standard/'
MODEL_DATA_GOOD = "/data/model/good/"
MODEL_DATA_BAD = "/data/model/bad/"
SPLIT_SYMBOL = '/'

# standard resource folder contains only measurement with 6 sensors
SENSORS_STANDARD = ["Прополис", "МУНТ", "ПЭГ-2000", "ТОФО", "Тритон", "ДЦГ-18К6", "ПЭГС", "ПФЭ", "ПЭГСб", "ПЭГФ"]

MODEL_DATE = ['29.04.16', '03.05.16-4', '03.05.16-6', '04.05.16']
SUBSTANCE_GOOD = ['acetaldehyde', 'ethanol', 'propanol-1', 'propanol-2']
SUBSTANCE_BAD = ['acetone', 'butanol-1', 'butanol-2', 'MEK']
MEASUREMENT = ['measurement-1.xlsx', 'measurement-2.xlsx', 'measurement-3.xlsx', 'measurement-4.xlsx']

DATE = ['13.05.16', '14.05.16', '16.05.16']
STANDARD_SUBSTANCE = ["1", "2", "5", "6", "12", "14"]
NON_STANDARD_SUBSTANCE = ["13", "15", "16"]


# parse standard measurement
def parser(path_to_file):
    START_COLUMN_MODEL = 6
    try:
        workbook = xlrd.open_workbook(path_to_file)
        sensors = get_list_of_sensors(path_to_file)
    except IOError:
        return
    values = []
    arr = []

    for s in SENSORS_STANDARD:
        if s in sensors:
            index = sensors.index(s)
            for elem in workbook.sheet_by_index(0).col(START_COLUMN_MODEL + index):
                values.append(elem.value)

            last_value = values[-1]
            # Neural network take a matrix with 120 rows so it is necessary to add additional rows with last_value value
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

    return np.transpose(arr)


def get_model_measurement(date, substance, measurement, is_good):
    START_NAME = 16
    if is_good:
        file_name = PATH_TO_DIR + MODEL_DATA_GOOD + MODEL_DATE[date] + SPLIT_SYMBOL + SUBSTANCE_GOOD[substance] \
                    + SPLIT_SYMBOL + MEASUREMENT[measurement]
    else:
        file_name = PATH_TO_DIR + MODEL_DATA_BAD + MODEL_DATE[date] + SPLIT_SYMBOL + SUBSTANCE_BAD[substance] \
                    + SPLIT_SYMBOL + MEASUREMENT[measurement]

    model = parser(file_name)
    # create_image_from_array(model, True, DATE[date], SUBSTANCE[substance], MEASUREMENT[measurement][-6])
    # print(model)
    print '\nFILE NAME: ' + file_name[START_NAME:]
    return model


def get_st_not_st_measurement(date, substance, measurement, is_standard):
    START_NAME = 0
    if is_standard:
        file_name = PATH_TO_DIR + STANDARD_DATA + DATE[date] + SPLIT_SYMBOL + STANDARD_SUBSTANCE[substance] \
                    + SPLIT_SYMBOL + MEASUREMENT[measurement]
    else:
        file_name = PATH_TO_DIR + NON_STANDARD_DATA + DATE[date] + SPLIT_SYMBOL + NON_STANDARD_SUBSTANCE[substance] \
                    + SPLIT_SYMBOL + MEASUREMENT[measurement]

    d = parser(file_name)
    print '\nFILE NAME: ' + file_name[START_NAME:]
    return d


def get_list_of_sensors(path_to_file):
    values = []
    workbook = xlrd.open_workbook(path_to_file)

    SENSORS_COLUMN = 3
    for elem in workbook.sheet_by_index(0).col(SENSORS_COLUMN):
        s = elem.value.encode('utf-8').strip()
        if s == "":
            break
        values.append(s)
    return values


# reformat from shape (m x n) to (1 x (m*n)), where first m line is 1 column, second m line is 2 column and etc.
def reformat_measurement(measurement):
    result = np.array(measurement[:, 0])

    for i in range(1, measurement.shape[1]):
        column = measurement[:, i]
        result = np.concatenate((result, column))
    return result


def get_model_data(is_good, is_reformat=False):
    data = []

    if is_good:
        substance = SUBSTANCE_GOOD
    else:
        substance = SUBSTANCE_BAD

    for d in range(0, len(MODEL_DATE)):
        print MODEL_DATE[d]
        for s in range(0, len(substance)):
            for m in range(0, len(MEASUREMENT)):
                # 03.05.16-4 contains only 2 measurements for each substance,
                # so we need to skip 3-rd measurement
                # if DATE[d] == DATE[1] and m == 2:
                #     continue
                st = get_model_measurement(d, s, m, is_good)
                if st is None:
                    continue
                elif is_reformat:
                    data.append(reformat_measurement(st))
                else:
                    data.append(st)
    return data


def get_all_st_non_st_data(is_standard, is_reformat=False):
    data = []

    if is_standard:
        substance = STANDARD_SUBSTANCE
    else:
        substance = NON_STANDARD_SUBSTANCE

    for d in range(0, len(DATE)):
        print DATE[d]
        for s in range(0, len(substance)):
            for m in range(0, len(MEASUREMENT)):
                mes = get_st_not_st_measurement(d, s, m, is_standard)

                if mes is None:
                    continue
                elif is_reformat:
                    data.append(reformat_measurement(mes))
                else:
                    data.append(mes)
    return data


# good_model = get_model_data(True)
# bad_model = get_model_data(False)

# standard = get_all_st_non_st_data(True)
# non_standard = get_all_st_non_st_data(False)


# print "Good model: " + str(len(good_model))
# print "Bad model: " + str(len(bad_model))
# print "Standard: " + str(len(standard))
# print "Non standard: " + str(len(non_standard))
