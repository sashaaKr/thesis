import os
import text_cleanup.text_cleanup as thesisCleanUp

A_ZWICKAU_FILE_NAME = "A_Zwickau_RB_I_XII_5 (12).txt"
B_LONDON_FILE_NAME = "B_London_BL_Add_18929 (6).txt"
STOP_WORDS_FILE_NAME = "stop_words.json"

USER_HOME_DIR = os.path.expanduser('~')
ROOT = os.path.join(USER_HOME_DIR, 'thesis',) 

def get_data_file_path(file_name):
    return os.path.join(ROOT, 'full', file_name)

a_zwickau_file_path = get_data_file_path(A_ZWICKAU_FILE_NAME)
b_london_file_path = get_data_file_path(B_LONDON_FILE_NAME)
stop_words_file_path = get_data_file_path(STOP_WORDS_FILE_NAME)

def read_file(file_path):
    return open(file_path, encoding='utf-8').read()

def read_zwickau():
    return read_file(a_zwickau_file_path)

def read_london():
    return read_file(b_london_file_path)

def get_london_corpus():
    london_text = read_london()
    london_corpus = thesisCleanUp.create_corpus_by_line(thesisCleanUp.jvtext(london_text))
    return london_corpus

def get_zwickau_corpus():
    zwickau_text = read_zwickau()
    zwickau_corpus = thesisCleanUp.create_corpus_by_line(thesisCleanUp.jvtext(zwickau_text))
    return zwickau_corpus