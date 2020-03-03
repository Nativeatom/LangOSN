import pandas as pd
import pickle
import string
import os
import re
import pdb

def get_batches(train, target, batch_size, word2index, padding_mode='max', level='character', to_yield=True, return_ratio=False):
    nums = int(len(train)/batch_size)
    batches = []
    correct_num = 0
    misspell_num = 0
    if level == 'character':
        for i in range(nums):
            train_batch = [sent2index(' '.join(x).split(' '), word2index) for x in train[i*batch_size: (i+1)*batch_size]]
            target_batch = [sent2index(' '.join(x).split(' '), word2index) for x in target[i*batch_size: (i+1)*batch_size]]
            if padding_mode == 'No':
                pass
            else:
                if padding_mode == 'max':
                    train_max = max([len(x) for x in train_batch])
                    target_max = max([len(x) for x in target_batch])
                else:
                    train_max = target_max = padding_mode

                train_batch = [x+[word2index['<PAD>']]*(train_max - len(x)) for x in train_batch]
                target_batch = [x + [word2index['<PAD>']] * (target_max - len(x)) for x in target_batch]

            batches.append([train_batch, target_batch])

        if i*batch_size < len(train):
            train_batch = [sent2index(' '.join(x).split(' '), word2index) for x in train[i * batch_size:]]
            target_batch = [sent2index(' '.join(x).split(' '), word2index) for x in target[i * batch_size:]]
            if padding_mode == 'No':
                pass
            else:
                if padding_mode == 'max':
                    train_max = max([len(x) for x in train_batch])
                    target_max = max([len(x) for x in target_batch])
                else:
                    train_max = target_max = padding_mode

                train_batch = [x + [word2index['<PAD>']] * (train_max - len(x)) for x in train_batch]
                target_batch = [x + [word2index['<PAD>']] * (target_max - len(x)) for x in target_batch]

            batches.append([train_batch, target_batch])

    if level == 'token':
        for i in range(nums):
            train_batch = [sent2index(x, word2index) for x in
                           train[i * batch_size: (i + 1) * batch_size]]
            target_batch = [sent2index(x, word2index) for x in target[i * batch_size: (i + 1) * batch_size]]
            if padding_mode == 'No':
                pass
            else:
                if padding_mode == 'max':
                    train_max = max([len(x) for x in train_batch])
                    target_max = max([len(x) for x in target_batch])
                else:
                    train_max = target_max = padding_mode

                train_batch = [x + [word2index['<UNK>']]*(train_max - len(x)) for x in train_batch]
                target_batch = [x + [word2index['<UNK>']] * (target_max - len(x)) for x in target_batch]
            batches.append([train_batch, target_batch])

        if i*batch_size < len(train):
            train_batch = [sent2index(x.replace('', ' ').split(), word2index) for x in
                           train[i * batch_size:]]
            target_batch = [sent2index(x.replace('', ' ').split(), word2index) for x in
                            target[i * batch_size:]]
            if padding_mode == 'No':
                pass
            else:
                if padding_mode == 'max':
                    train_max = max([len(x) for x in train_batch])
                    target_max = max([len(x) for x in target_batch])
                else:
                    train_max = target_max = padding_mode

                train_batch = [x + [word2index['<UNK>']] * (train_max - len(x)) for x in train_batch]
                target_batch = [x + [word2index['<UNK>']] * (target_max - len(x)) for x in target_batch]
            batches.append([train_batch, target_batch])

    if level == 'binary':
        misspell = 1
        correct = 0
        for source_words, target_words in zip(train, target):
            source_words = [x.strip(string.punctuation) for x in source_words.split()]
            target_words = [x.strip(string.punctuation) for x in target_words.split()]
            for source_word, target_word in zip(source_words, target_words):
                source_idx = sent2index(source_word, word2index)
                target_idx = sent2index(target_word, word2index)
                pdb.set_trace()
                if source_idx == target_idx:
                    batches.append([source_idx, correct, target_idx])
                    correct_num += 1
                else:
                    if len(source_idx) > 2 and len(target_idx) > 2:
                        batches.append([source_idx, misspell, target_idx])
                        misspell_num += 1
                        if sent2index(source_word, word2index) == sent2index(target_word, word2index):
                            pdb.set_trace()

    if level == 'dict':
        pass

    batches = [x for x in batches if len(x[0]) > 0]
    print('correct: {}({}%) misspelled: {}({}%)'.format(correct_num, round(correct_num*100 / (misspell_num + correct_num), 2),
                                                        misspell_num, round(misspell_num*100 / (misspell_num + correct_num), 2)))
    for batch in batches:
        yield batch

def sent2index(sent, word2index):
    result = []
    first_PAD = True
    for id, x in enumerate(sent):
        if id != len(sent) - 1:
            if x != "<PAD>":
                result.append(word2index.get(x, word2index['<UNK>']))
            else:
                if first_PAD:
                    result.append(word2index['<EOS>'])
                    first_PAD = False
                else:
                    result.append(word2index['<PAD>'])
    # pdb.set_trace()
    return result
    # return [word2index.get(x, word2index['<UNK>']) for x in sent]


def summary_saver(summary):
    summary_file = 'summary.csv'
    columns = list(summary.keys())
    if not os.path.isfile(summary_file):
        summary_data = pd.DataFrame(columns=columns)
    else:
        summary_data = pd.read_csv(summary_file, encoding='utf-8')

    summary_data = summary_data.loc[:, ~summary_data.columns.str.contains('^Unnamed')]
    summary_data = summary_data.append(summary, ignore_index=True)
    summary_data.to_csv(summary_file, encoding='utf-8')
    print('Summary saved at ', summary_file)

def dict_saver(dic, name):
    with open(name, 'wb') as fp:
        pickle.dump(dic, fp)
    fp.close()

def preprocess(x):
    return re.sub('[%s]' % re.escape('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~-'), '', x)
