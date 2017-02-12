import parser

array = parser.get_standard_measurement(1, 5, 2)

file_list = parser.get_list_of_files_in_dir(parser.PATH_TO_DIR, False)
other = parser.get_other_measurement(file_list[3])

parser.create_image_from_array(array)
