import parser

for d in range(0, len(parser.DATE)):
    for s in range(0, len(parser.SUBSTANCE)):
        for m in range(0, len(parser.MEASUREMENT)):
            parser.get_standard_measurement(d, s, m)


# array = parser.get_standard_measurement(1, 5, 2)

# file_list = parser.get_list_of_files_in_dir(parser.PATH_TO_DIR, False)
# other = parser.get_other_measurement(file_list[3])

# parser.create_image_from_array(array)
