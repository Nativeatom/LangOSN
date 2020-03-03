from config import *
import string

def load_data(lang, frequent_num):
    if lang == 'English_normal':
        with open('./files/eng_TOEFL/highrate/en_most_frequent_{}_table.txt'.format(frequent_num), 'r') as fp:
            train = fp.readlines()
            train = [x.strip('\n').split() for x in train]
        fp.close()

        # dev set (shares 27% vocabulary with training set)
        with open('./files/eng_TOEFL/highrate/en_most_frequent_50_in_whole_oov_200.txt', 'r') as fp:
            dev = fp.readlines()
            dev = [x.strip('\n').split() for x in dev]
        fp.close()

        # test set
        with open('./files/eng_TOEFL/highrate/en_most_frequent_200_in_whole_oov_200.txt', 'r') as fp:
            test = fp.readlines()
            test = [x.strip('\n').split() for x in test]
        fp.close()

        # test set (oov_1 and oov_2 shares 10% in vocabulary)
        with open('./files/eng_TOEFL/highrate/en_most_frequent_200_in_whole_oov_200_2.txt', 'r') as fp:
            test2 = fp.readlines()
            test2 = [x.strip('\n').split() for x in test2]
        fp.close()

        # pure correct set
        with open("./files/eng_TOEFL/highrate/en_most_frequent_200_in_whole_oov_200_pure_correct.txt", "r") as fp:
            pure_correct = fp.readlines()
            pure_correct = [x.strip('\n').split() for x in pure_correct]
        fp.close()

        # byrule set
        with open("./files/English/city/New_York_City_byrule_label_200_position_test_set.txt", "r") as fp:
            byrule_set = fp.readlines()
            byrule_set = [x.strip('\n').strip(string.punctuation).split() for x in byrule_set]
        fp.close()

    else:
        langcode = lang2code[lang]
        with open('./files/{}/city/{}_most_frequent_{}_table.txt'.format(lang, langcode, frequent_num), 'r') as fp:
            train = fp.readlines()
            train = [x.strip('\n').split() for x in train]
            train = [x for x in train if x[0] != '\u200b' and len(x.split()) > 0]
        fp.close()

        # dev set (shares 27% vocabulary with training set)
        with open('./files/{}/city/{}_most_frequent_50_in_whole_oov_200.txt'.format(lang, langcode), 'r') as fp:
            dev = fp.readlines()
            dev = [x.strip('\n').split() for x in dev]
        fp.close()

        # test set
        with open('./files/{}/city/{}_most_frequent_200_in_whole_oov_200.txt'.format(lang, langcode), 'r') as fp:
            test = fp.readlines()
            test = [x.strip('\n').split() for x in test]
            test = [x for x in test if x[0]!='k²']
        fp.close()

        with open('./files/{}/city/{}_most_frequent_200_in_whole_oov_200_pure_correct.txt'.format(lang, langcode), "r") as fp:
            pure_correct = fp.readlines()
            pure_correct = [x.strip('\n').split() for x in pure_correct]
        fp.close()

    return train, dev, test, test2, pure_correct
