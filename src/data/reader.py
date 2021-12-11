import os

A_ZWICKAU_FILE_NAME = "A_Zwickau_RB_I_XII_5 (12).txt"
B_LONDON_FILE_NAME = "B_London_BL_Add_18929 (6).txt"
STOP_WORDS_FILE_NAME = "stop_words.json"

ROOT = os.path.join('..') # root related to notebook

def get_data_file_path(file_name):
    return os.path.join(ROOT, 'full', file_name)

a_zwickau_file_path = get_data_file_path(A_ZWICKAU_FILE_NAME)
b_london_file_path = get_data_file_path(B_LONDON_FILE_NAME)
stop_words_file_path = get_data_file_path(STOP_WORDS_FILE_NAME)

def read_file(file_path):
    return open(file_path, encoding='utf-8').read()

def read_a_zwickau():
    return read_file(a_zwickau_file_path)

def read_b_london():
    return read_file(b_london_file_path)