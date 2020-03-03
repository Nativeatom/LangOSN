from config import *
import numpy as np
from Useful_Function import character_preprocessing

train_flation_statistics = {}
train_flation_statistics_log = {}

ratio_flation = False

# default 0.00001
log_epsilon = 1e-06

train_volumes = [50] + [x for x in np.arange(100, 3000, 100)]
langs = ['English', 'Italian', 'Spanish', 'Russian', 'Griko', 'Turkish', 'Finnish']

for lang in langs:
    if lang == 'eng_TOEFL':
        langcode = 'en'
        lang_path_addition = ''
    elif lang == 'English':
        langcode = lang2code[lang]
        lang_path_addition = '/USCities'
    else:
        langcode = lang2code[lang]
        lang_path_addition = ''

    train_flation_statistics[lang] = {'sumRatio': [], 'minimumTime': [], 'volume': [], 'RealVolume': []}
    train_flation_statistics_log[lang] = {'sumRatio': [], 'minimumTime': [], 'volume': [], 'RealVolume': []}
    if lang != 'English':
        lpa = ''
    else:
        lpa = lang_path_addition

    langCode = lang2code[lang]
    for train_volume in train_volumes:
        try:
            with open('./files/{}/city{}/high_frequent/{}_most_frequent_{}_table.txt'.format(lang, lpa,
                                                                                             langCode, train_volume),
                      'r') as fp:
                train = fp.readlines()
                train = [x.strip('\n').split() for x in train]
                train = [x for x in train if '\u200b' or '°' not in x[0] and len(x) > 0]
            fp.close()

            train_words = [character_preprocessing(x[0]) for x in train]
            train_tags = [0 for x in train]
            train_appearance = [int(x[1]) for x in train]

            # based on frequency
            train_freqs = [float(x[2]) for x in train]

            # based on appearance
            train_freqs_log = [np.log(x) for x in train_appearance]

            train_flation_statistics[lang]['sumRatio'].append(sum(train_freqs))
            train_flation_statistics_log[lang]['sumRatio'].append(sum(train_freqs_log))

            frequency_training_min_num = int(1 / (train_freqs[-1] + log_epsilon)) + 1

            train_flation_statistics[lang]['minimumTime'].append(frequency_training_min_num)

            freq_training_corpus = []
            freq_training_corpus_log = []

            for word, freq, freq_log in zip(train_words, train_freqs, train_freqs_log):
                freq_training_corpus += [word for _ in range(int(frequency_training_min_num * freq))]
                freq_training_corpus_log += [word for _ in range(int(freq_log) + 1)]

            train_flation_statistics[lang]['RealVolume'].append(sum(train_appearance))
            train_flation_statistics[lang]['volume'].append(len(freq_training_corpus))
            train_flation_statistics_log[lang]['volume'].append(len(freq_training_corpus_log))

            # write to file
            with open('./files/{}/city{}/high_frequent/{}_most_frequent_{}_table_corpus.txt'.format(lang, lpa,
                                                                                            langCode,
                                                                                            train_volume), 'w') as fp:
                for word in freq_training_corpus:
                    fp.write(' '.join([word, '0', '0', 'normal']))
                    fp.write('\n')

            with open('./files/{}/city{}/high_frequent/{}_most_frequent_{}_table_corpus_log.txt'.format(lang, lpa,
                                                                                                langCode,
                                                                                                train_volume),'w') as fp:
                for word in freq_training_corpus_log:
                    fp.write(' '.join([word, '0', '0', 'normal']))
                    fp.write('\n')


        except Exception as e:
            print(lang, train_volume, e)
    print("Finish {}".format(lang))