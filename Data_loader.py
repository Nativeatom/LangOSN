import codecs
import string
import re
from config import ascii_lowercase, glagolitsa, letters


def DataLoader(lang, letters):
    # load training and test file by language
    lang2file = {
        'fin': {'train': './files/Finnish/BlackSabbath_train.txt', 'test': './files/Finnish/artif_all_indomain_test_Beatles.txt'},
        'ita': {'train': './files/Italian/BlackSabbath_train.txt', 'test': './files/Italian/artif_Italian_test_Beatles.txt'},
        'spa': {'train': './files/Spanish/BlackSabbath_train.txt', 'test': './files/Spanish/artif_Spanish_test_Beatles.txt'},
        'ru_btl': {'train': './files/Russian/BlackSabbath_train.txt', 'test': './files/Russian/artif_Russian_test_Beatles.txt'}, 
        'trk': {'train': './files/Turkish/BlackSabbath_train.txt', 'test': './files/Turkish/artif_Turkish_test_Beatles.txt'},
        'fin_wiki_rule': {'train': './files/Finnish/politician/Carl Gustaf Emil Mannerheim byrule.txt', 'test': './files/Finnish/politician/Jean Sibelius byrule.txt'},
        'ita_wiki_rule': {'train': './files/Italian/politician/Silvio Berlusconi byrule.txt', 'test': './files/Italian/politician/Gaio Giulio Cesare byrule.txt'},
        'spa_wiki_rule': {'train': './files/Spanish/language/Idioma español byrule.txt', 'test': './files/Spanish/language/Idioma catalán byrule.txt'},
        'ru_wiki_rule': {'train': './files/Russian/politician/Горбачёв, Михаил Сергеевич byrule.txt', 'test': './files/Russian/history/Союз Советских Социалистических Республик byrule.txt'},
        'trk_wiki_rule': {'train': './files/Turkish/politician/Mustafa Kemal Atatürk byrule.txt', 'test': './files/Turkish/politician/Recep Tayyip Erdoğan byrule.txt'}
    }

    with codecs.open(lang2file[lang]['train'], 'r', 'utf-8') as fp:
        train_raw = fp.readlines()
        train_raw = [x.strip('\n').strip('\r') for x in train_raw]
        train_raw = [x for x in train_raw if not x.strip(string.punctuation).isdigit()]

    with codecs.open(lang2file[lang]['test'], 'r', 'utf-8') as fp:
        Target_test = fp.readlines()
        Target_test = [x.strip('\n').strip('\r') for x in Target_test]
        Target_test = [x for x in Target_test if not x.strip(string.punctuation).isdigit()]

    pure_letters = ' '.join(ascii_lowercase).split()
    if lang in ['ru_btl', 'ru_wiki_rule']:
        # Using Cyrillic alphabet for Russian
        letters = glagolitsa

    i2v = {i + 1: letters[i] for i in range(len(letters))}
    i2v[0] = '<PAD>'
    i2v[len(i2v)] = ' '
    i2v[len(i2v)] = '-'
    i2v[len(i2v)] = '<UNK>'
    i2v[len(i2v)] = '<EOS>'
    i2v[len(i2v)] = '<SOS>'
    v2i = {v: k for k, v in i2v.items()}

    return train_raw, Target_test, test_file, i2v, v2i, pure_letters, letters