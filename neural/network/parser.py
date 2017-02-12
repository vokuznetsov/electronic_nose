# coding=utf-8
import xlrd
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image

PATH_TO_DIR = './resource'
SPLIT_SYMBOL = '/'

# standard resource folder contains only measurement with 6 sensors
SENSORS_STANDARD = ["Прополис", "МУНТ", "ПЭГ-2000", "ТОФО", "Тритон", "ДЦГ-18К6"]

DATE = ['29.04.16', '03.05.16']
SUBSTANCE = ['acetaldehyde', 'acetone', 'butanol-1', 'butanol-2', 'ethanol', 'MEK',
             'propanol-1', 'propanol-2', 'toluene']
MEASUREMENT = ['measurement-1.xlsx', 'measurement-2.xlsx', 'measurement-3.xlsx']


# parse standard measurement
def parser(path_to_file, is_standard):
    START_COLUMN_STANDARD = 6
    START_COLUMN_OTHER = 1
    START_ROW_OTHER = 10
    workbook = xlrd.open_workbook(path_to_file)
    sensors = get_list_of_sensors(path_to_file, is_standard)
    values = []
    arr = []
    count = 0

    for s in SENSORS_STANDARD:
        if s in sensors:
            index = sensors.index(s)

            if is_standard:
                for elem in workbook.sheet_by_index(0).col(START_COLUMN_STANDARD + index):
                    values.append(elem.value)
            else:
                for elem in workbook.sheet_by_index(0).col(START_COLUMN_OTHER + index):
                    count += 1
                    if count > START_ROW_OTHER:
                        values.append(elem.value)
        else:
            # so we should have a matrix with size 119x8, we should add a zeros vector with size (119x1)
            # zeros vectors because the sensors is not contained in SENSORS_STANDARD
            if is_standard:
                values = np.zeros(119)
            else:
                values = np.zeros(61)

        if len(arr) == 0:
            arr = np.array(values)
        else:
            arr = np.vstack((arr, values))
        values = []
        count = 0

    return np.transpose(arr)


def get_standard_measurement(date, substance, measurement):
    START_NAME = 29
    file_name = PATH_TO_DIR + '/standard/' + DATE[date] + SPLIT_SYMBOL + SUBSTANCE[substance] \
                + SPLIT_SYMBOL + MEASUREMENT[measurement]
    standard = parser(file_name, is_standard=True)
    print(standard)
    print '\nFILE NAME: ' + file_name[START_NAME:]
    return standard


def get_other_measurement(file_name):
    path_to_file = PATH_TO_DIR + SPLIT_SYMBOL + file_name
    other = parser(path_to_file, is_standard=False)
    print('\n' + str(other))
    print '\nFILE NAME: ' + path_to_file
    return other


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


def create_image_from_array(array):
    # array = np.zeros((6, 119))
    # values = []
    #
    # for i in range(0, 119):
    #     values.append(200.0)
    #
    # array = np.vstack((array, values))
    formatted = (array * 255 / np.max(array)).astype('uint8')

    img = Image.fromarray(formatted, 'L')
    img.save('./visualization/my.png')
    # img.show()

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
