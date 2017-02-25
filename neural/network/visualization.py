import parser


standard_data = parser.get_all_standard_data()
non_stadard_data = parser.get_all_non_standard_data()

print len(standard_data)
print len(non_stadard_data)

# array = parser.get_standard_measurement(1, 5, 2)

# file_list = parser.get_list_of_files_in_dir(parser.PATH_TO_DIR, False)
# other = parser.get_other_measurement(file_list[3])

# parser.create_image_from_array(array)
