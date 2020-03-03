import numpy as np
try:
    from scipy.misc import logsumexp
except:
    from scipy.special import logsumexp
import random
import string
import time
from datetime import datetime, timedelta
from collections import defaultdict
import re
from config import *
import pdb

choice = ['l2ll', 'll2l', 'r2rr', 'rr2r', 'l2r', 'n2nn', 'delete vowel', 'double vowel', 'c/s/z', 'j/g',
          'remove slient cons',
          'f2ph', 'p2b', 's2c', 'e2a', 'b2v', 'r2d', 'v2b', 'partialdrop', 'insert j', 'drop h']

rule2id = {r:i for i, r in enumerate(choice)}
id2rule = {v:k for k, v in rule2id.items()}

def extract_train_volume(name):
    try:
        pattern = re.compile(r'train\[(.*?)\]', re.S)
        return re.findall(pattern, name)[0]
    except:
        return '0'


def get_timestamp():
    time_stamp = int(time.time())
    loc_time = time.localtime(time_stamp)
    time1 = time.strftime("%Y-%m-%d %H:%M:%S",loc_time)
    utc_time = datetime.utcfromtimestamp(time_stamp)
    return utc_time.isoformat()

def extract_timestamp(name):
    try:
        pattern = re.compile(r'date\[(.*?)\]', re.S)
        return re.findall(pattern, name)[0].replace("#", ":")
    except:
        # return current time
        logger.info("fail to extract timestamp from {}".format(name))
        return get_timestamp()

def softmax(x, axis):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

def misspell_clean(data):
    to_remove = ["`` 1 of", "'' 1 of", "'s 1 is", "40 1 of", "50 1 of", "n't 1 not",
                 "20 1 of", "18th 1 with", "'m 1 am", ".. 1 of", "'d 1 d"]
    return [x for x in data if x not in to_remove]

def get_grad(variable):
    print(variable)

def misspelling_generator_wrapper(word, letters, times, pattern='none'):
    options = ['delete', 'transpose', 'replace', 'insert']
    for i in range(times):
        action = random.choice(options)
        while action == 'delete' and len(word) == 1:
            action = random.choice(options)
        err_word, position, action = misspelling_generator(word, letters, action, pattern)
    return err_word, position, action


def misspelling_generator(word, letters, action, pattern='none'):
    length = len(word)
    if pattern == 'none':
        err_word = 'none'

        if action == 'delete':
            position = random.choice(range(length))
            err_word = word[:position] + word[position + 1:]

        if action == 'transpose':
            if length > 1:
                position = random.choice(range(length - 1))
                err_word = word[:position] + word[position + 1] + word[position] + word[position + 1:]
            else:
                err_word, position, action = misspelling_generator(word, letters, 'replace')

        if action == 'insert':
            position = random.choice(range(length))
            insertion = random.choice(letters)
            err_word = word[:position] + insertion + word[position:]

        if action == 'replace':
            position = random.choice(range(length))
            insertion = random.choice(letters)
            err_word = word[:position] + insertion + word[position + 1:]

        return err_word, position, action

    else:
        err_word, position, action = misspelling_generator_pattern(word, letters, action, pattern)
    return err_word, position, action


def misspelling_generator_pattern(word, letters, action, pattern):
    err_produce = 0
    attempt = 0
    if pattern != 'ru':
        while not err_produce and attempt < 5:
            attempt += 1
            action = random.choice(rules[pattern])
            if '2' in action:
                prev, curr = action.split('2')
                if prev not in word:
                    continue
                err_word = word.replace(prev, curr)
                err_produce += 1
                position = len(prev)
                break

            if '/' in action:
                alternatives = action.split('/')
                for i, alternative in enumerate(alternatives):
                    if alternative in word:
                        replace = alternative
                        while (replace == alternative):
                            replace = random.choice(alternatives)
                        position = word.index(alternative)
                        err_word = word[:position] + replace + word[position + 1:]
                        err_produce += 1
                        break

            if action == 'delete_vowel':
                vowel_select = None
                word_len = len(word)
                iteration = 0
                while (vowel_select is None and iteration < min(5, word_len)):
                    vowel_index = random.choice(range(word_len))
                    if word[vowel_index] in vowels[pattern]:
                        vowel_select = word[vowel_index]
                        err_word = word[:vowel_index] + word[vowel_index + 1:]
                        err_produce += 1
                        position = vowel_index
                        break
                    iteration += 1

            if action == 'replace_vowel':
                vowel_select = None
                word_len = len(word)
                iteration = 0
                while(vowel_select is None and iteration < min(5, word_len)):
                    vowel_index = random.choice(range(word_len))
                    if word[vowel_index] in vowels[pattern]:
                        vowel_select = word[vowel_index]
                        vowel_replace = word[vowel_index]
                        while vowel_replace == vowel_select:
                            vowel_replace = random.choice(vowels[pattern])
                        err_word = word[:vowel_index] + vowel_replace + word[vowel_index + 1:]
                        err_produce += 1
                        position = vowel_index
                        break
                    iteration += 1

            if action == 'double_vowel':
                vowel_select = None
                word_len = len(word)
                iteration = 0
                while(vowel_select is None and iteration < min(5, word_len)):
                    vowel_index = random.choice((range(word_len)))
                    if word[vowel_index] in vowels[pattern]:
                        vowel_select = word[vowel_index]
                        err_word = word[:vowel_index] + vowel_select + word[vowel_index + 1:]
                        err_produce += 1
                        position = vowel_index
                        break
                    iteration += 1

            if action == 'drop_h':
                if 'h' in word:
                    position = word.index('h')
                    err_word = word[:position] + word[position + 1:]
                    err_produce += 1
                    break

            if action == 'insert_j':
                position = random.choice(range(len(word)))
                err_word = word[:position] + 'j' + word[position + 1:]
                err_produce += 1
                break

        if not err_produce:
            err_word, position, action = misspelling_generator_wrapper(word, letters, 1, pattern='none')

    if pattern == 'ru':
        while not err_produce and attempt < 5:
            attempt += 1
            action = random.choice(rules[pattern])
            if action == 'remove_slient_cons':
                if 'ы' in word:
                    position = word.index('ы')
                    err_word = word[:position] + word[position + 1:]
                    err_produce += 1
                    break

            if '2' in action:
                prev, curr = action.split('2')
                if Latin2Kiril.get(prev, 'none') not in word:
                    continue
                err_word = word.replace(Latin2Kiril[prev], Latin2Kiril[curr])
                err_produce += 1
                position = len(prev)
                break

            if '/' in action:
                alternatives = action.split('/')
                alternatives = [Latin2Kiril[x] for x in alternatives]
                for i, alternative in enumerate(alternatives):
                    if alternative in word:
                        replace = alternative
                        while (replace == alternative):
                            replace = random.choice(alternatives)
                        position = word.index(alternative)
                        err_word = word[:position] + replace + word[position + 1:]
                        err_produce += 1
                        break

            if action == 'delete_vowel':
                vowel_select = None
                word_len = len(word)
                iteration = 0
                while (vowel_select is None and iteration < min(5, word_len)):
                    start_index = random.choice(range(word_len))
                    for index, character in enumerate(word[start_index:]):
                        if character in vowels[pattern]:
                            vowel_index = start_index + index
                            if word[vowel_index] in vowels[pattern]:
                                vowel_select = word[vowel_index]
                                err_word = word[:vowel_index] + word[vowel_index + 1:]
                                err_produce += 1
                                position = vowel_index
                                break
                    iteration += 1

            if action == 'replace_vowel':
                vowel_select = None
                word_len = len(word)
                iteration = 0

                while vowel_select is None and iteration < min(5, word_len):
                    start_index = random.choice(range(word_len))
                    for index, character in enumerate(word[start_index:]):
                        if character in vowels[pattern]:
                            vowel_index = start_index + index
                    # if word[vowel_index] in vowels[pattern]:
                            vowel_select = word[vowel_index]
                            vowel_replace = word[vowel_index]
                            while vowel_replace == vowel_select:
                                vowel_replace = random.choice(vowels[pattern])
                            err_word = word[:vowel_index] + vowel_replace + word[vowel_index + 1:]
                            err_produce += 1
                            position = vowel_index
                            break
                    iteration += 1

            if action == 'double_vowel':
                vowel_select = None
                word_len = len(word)
                iteration = 0
                while (vowel_select is None and iteration < min(5, word_len)):
                    start_index = random.choice(range(word_len))
                    for index, character in enumerate(word[start_index:]):
                        if character in vowels[pattern]:
                            vowel_select = word[start_index+index]
                            err_word = word[:start_index+index] + vowel_select + word[start_index+index + 1:]
                            err_produce += 1
                            position = vowel_index
                            break
                    iteration += 1

        if not err_produce:
            err_word, position, action = misspelling_generator_wrapper(word, letters, 1, pattern='none')

    return err_word, position, action

def character_preprocessing(x):
    return x.lower().strip("”%"+string.punctuation).replace('/', ' ').replace('°', ''). \
                replace('—', '-').replace('º', '').replace('«', '').replace('»', ''). \
                replace('=', '').replace('[', ' '). \
                replace(']', ' ').replace('(', ' ').replace(')', ' ').replace(',', '').replace('.', ''). \
                replace('"', '').replace("'", '').replace('—', '').replace('km²', '').replace('′', ''). \
                replace(':', '').replace(';', '').replace('；', '').replace('≥', '').replace('&', '').\
                replace('‘', '').replace('ʻ', '').replace('•', '').replace('m²', '')


def Preprocess(filepath, edit_prob, lang, err=True, write2file=True, pattern='none'):
    if lang == 'Russian':
        letter_set = glagolitsa_pure
    else:
        letter_set = ' '.join(ascii_lowercase).split()

    with open(filepath, 'r') as fp:
        data = fp.readlines()
        data = [character_preprocessing(x.strip('\n')) for x in data]
        data = [x for x in data if len(x)]

    data = ' '.join(data).split()
    data_clean_indomain = [x for x in data if x != '—' and not x.isdigit() and x not in string.punctuation]
    if err:
        data_mixed = []
        indexes = list(range(len(data_clean_indomain)))
        random.shuffle(indexes)
        threshold = int(edit_prob * len(indexes))
        error_indexes = indexes[:threshold]
        for index, word in enumerate(data_clean_indomain):
            if len(word) == 0:
                continue
            if index in error_indexes:
                err_word, position, action = misspelling_generator_wrapper(word.lower(), letter_set, 1, pattern)
                if len(err_word) > 0 and err_word != word:
                    if action == 'c/s/z':
                        data_mixed.append([err_word, '1', word, str(position), 'c_s_z'])
                    elif action == 'j/g':
                        data_mixed.append([err_word, '1', word, str(position), 'j_g'])
                    else:
                        data_mixed.append([err_word, '1', word, str(position), action])
            else:
                data_mixed.append([word, '0', word, '0', 'origin'])

        if write2file:
            with open(filepath.rstrip('.txt') + '_byrule.txt', 'w') as fp:
                for x in data_mixed:
                    fp.write(' '.join(x) + '\n')
            fp.close()

    else:
        data_mixed = [[word, str(0), word] for word in data_clean_indomain]

    return data_mixed


def Unit_test(lang='ru'):
    if lang == 'ru':
        # Russian file
        word = 'союз'
        err, pos, act = misspelling_generator_wrapper(word, glagolitsa_pure, 1, pattern='none')
        pdb.set_trace()

        filepath = './files/Russian/city/Санкт-Петербург.txt'
        lang = 'Russian'
        prob = 0.1
        data = Preprocess(filepath, prob, lang, err=True, write2file=False, pattern='ru')

    if lang == 'tr':
        filepath = './files/Turkish/history/Selçuklu Hanedanı.txt'
        lang = 'Turkish'
        prob = 0.1
        data = Preprocess(filepath, prob, lang, err=True, write2file=False, pattern='tr')

    pdb.set_trace()
    return data

if __name__ == "__main__":
    import os
    prob = 0.1
    lang_set = ['Russian']
    for lang in lang_set:
        path = './files/' + lang
        if os.path.isdir(path):
            for catgry in os.listdir(path):
                dirpath = path + '/' + catgry + '/'
                if os.path.isdir(dirpath):
                    for file in os.listdir(dirpath):
                        if file.endswith(".txt") and not file.endswith(" clean.txt"):
                            filepath = dirpath + file
                            data = Preprocess(filepath, prob, lang, err=True, write2file=True, pattern=lang2code[lang])
        print("Finish " + lang)




