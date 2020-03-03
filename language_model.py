import os
import sys
import itertools
import numpy as np
from collections import Counter
from evaluation import metrics_cal
# import matplotlib.pyplot as plt
import pdb
import logging
logging.basicConfig(level=logging.INFO,
                    filename='./log/language_model_training.txt',
                    filemode='a+',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
consoleHandler = logging.StreamHandler(sys.stdout)
logger.addHandler(consoleHandler)

def ngram_model(docs, n=2, pad_left=False, pad_right=False,
                left_pad_symbol=None, right_pad_symbol=None, padding='<PAD>'):
    history = []
    ngram2index = {}
    trigramProb = {}
    bigramProb = {}
    vocab = []

    # left pad enough grams to make the first letter of ngram
    docs_ngrams = itertools.chain([[left_pad_symbol] * pad_left * max(n-1, 1) + doc + [right_pad_symbol] * pad_right for doc in docs])

    for doc_ngrams in docs_ngrams:
        doc_history = []
        doc_ngrams = doc_ngrams + [padding] * (n - len(doc_ngrams)) if len(doc_ngrams) < n else doc_ngrams
        for i in range(len(doc_ngrams) - n + 1):
            ngram = doc_ngrams[i:i + n]
            doc_history.append(' '.join(ngram))
            trigramProb[' '.join(ngram)] = trigramProb.get(' '.join(ngram), 0) + 1

            # only include the first bigram (* <EOS> is not included)
            bigramProb[' '.join(ngram[:-1])] = bigramProb.get(' '.join(ngram[:-1]), 0) + 1
            # bigramProb[' '.join(ngram[1:])] = bigramProb.get(' '.join(ngram[1:]), 0) + 1
        history.append(doc_history)
        vocab += doc_ngrams

    trigramProb['vol'] = sum(trigramProb.values())
    bigramProb['vol'] = sum(bigramProb.values())
    vocab = list(set(vocab))

    return history, trigramProb, bigramProb, vocab

def ngram_model_update(docs, n=2, pad_left=False, pad_right=False,
                left_pad_symbol=None, right_pad_symbol=None, padding='<PAD>', ngramProbs={2:{}, 3:{}, 'vocab':[]}):
    history = []
    ngram2index = {}
    trigramProb = ngramProbs[3]
    bigramProb = ngramProbs[2]
    vocab = ngramProbs['vocab']

    # left pad enough grams to make the first letter of ngram
    docs_ngrams = itertools.chain([[left_pad_symbol] * pad_left * max(n-1, 1) + doc + [right_pad_symbol] * pad_right for doc in docs])

    for doc_ngrams in docs_ngrams:
        doc_history = []
        doc_ngrams = doc_ngrams + [padding] * (n - len(doc_ngrams)) if len(doc_ngrams) < n else doc_ngrams
        for i in range(len(doc_ngrams) - n + 1):
            ngram = doc_ngrams[i:i + n]
            doc_history.append(' '.join(ngram))
            trigramProb[' '.join(ngram)] = trigramProb.get(' '.join(ngram), 0) + 1

            # only include the first bigram (* <EOS> is not included)
            bigramProb[' '.join(ngram[:-1])] = bigramProb.get(' '.join(ngram[:-1]), 0) + 1
            # bigramProb[' '.join(ngram[1:])] = bigramProb.get(' '.join(ngram[1:]), 0) + 1
        history.append(doc_history)
        vocab += doc_ngrams

    trigramProb['vol'] = sum(trigramProb.values())
    bigramProb['vol'] = sum(bigramProb.values())
    vocab = list(set(vocab))

    return history, trigramProb, bigramProb, vocab


def text2ngram(docs, n=2, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None, padding='<PAD>'):
    # docs should be in form [['n','i'], ['h', 'a', 'o']]
    history = []

    if len(docs) == 0:
        return history

    # left pad enough grams to make the first letter of ngram
    docs_ngrams = itertools.chain([[left_pad_symbol] * pad_left * max(n-1, 1) + doc + [right_pad_symbol] * pad_right for doc in docs])
    for doc_ngrams in docs_ngrams:
        doc_history = []
        doc_ngrams = doc_ngrams + [padding] * (n - len(doc_ngrams)) if len(doc_ngrams) < n else doc_ngrams
        for i in range(len(doc_ngrams) - n + 1):
            ngram = doc_ngrams[i:i + n]
            doc_history.append(' '.join(ngram))
        history.append(doc_history)
        # vocab += doc_ngrams

    return history


def normalize_ngramProb(prob):
    prob_normalized = {}
    vol = prob['vol']
    for key, value in prob.items():
        prob_normalized[key] = value / vol
    return prob_normalized


def languageModelTrain(tokens_tags, ngram=3, smoothing=True, k=1, lengthAverage=True,
                       metric='f1', criterion='gridSearch', interval=0.05,
                       show_result=True, return_likelihood=False, setting={'setting':'train', 'ngramInfo': {2:{}, 3:{}, 'vocab':[]}, 'loglikelihoods':[], 'tokens':[], 'tags':[]}):
    tokens = tokens_tags
    try:
        tokens_train = list(itertools.chain([' '.join(x[0]).split() for x in tokens]))
        tags_train = list(itertools.chain([int(x[1]) for x in tokens]))
    except:
        pdb.set_trace()

    # Language Model must calculate all the loglikelihood again (unless in trading off calculation and approximation)            
    loglikelihoods = setting['loglikelihoods']
    print('length of loglikelihoods: ', len(loglikelihoods))
    # print("train setting: {}".format(setting['setting']))
    if setting['setting'] == 'train':
        tokens_trigram, trigramProb, bigramProb, vocab = ngram_model(tokens_train, n=ngram, pad_left=True, pad_right=True,
                                                        left_pad_symbol="<SOS>", right_pad_symbol="<EOS>")
        V = len(vocab)
    elif setting['setting'] == 'update':
        tokens_trigram, trigramProb, bigramProb, vocab = ngram_model_update(tokens_train, n=ngram, pad_left=True, pad_right=True,
                                                    left_pad_symbol="<SOS>", right_pad_symbol="<EOS>", ngramProbs=setting['ngramInfo'])
        # combine with the ngram of the original corpus
        corpus_ngrams = text2ngram([[x for x in y] for y in setting['tokens']], n=ngram, pad_left=True, pad_right=True, left_pad_symbol="<SOS>", right_pad_symbol="<EOS>")
        tokens_trigram += corpus_ngrams
        tags_train += setting['tags']
        vocab = setting['ngramInfo']['vocab']
        V = len(vocab)

    for token_trigrams in tokens_trigram:
        token_score = 0

        for length, trigram in enumerate(token_trigrams):
            # \lambda log((C(w1w2w3) + 1)/(C(w1w2) + ||V||))
            # token_score += np.log(
            #         (trigramProb.get(trigram, 0) + smoothing) / (
            #         bigramProb.get(' '.join(trigram.split()[:-1]), 0) + bigramProb['vol']))
            token_score += np.log(
            (trigramProb.get(trigram, 0) + k*smoothing) / (
            bigramProb.get(' '.join(trigram.split()[:-1]), 0) + k*V))

        if lengthAverage:
            loglikelihoods.append(token_score / (length + 1))
        else:
            loglikelihoods.append(token_score)

    thresholds_acc = []
    thresholds_f1 = []
    best_threshold_acc = [0, 0]
    best_threshold_f1 = [0, 0]

    # choose the threshold that will have the best F1
    for threshold_sample in np.arange(np.min(loglikelihoods), np.max(loglikelihoods), interval):
        predicts_lm = [int(x <= threshold_sample) for x in loglikelihoods]
        acc, precision, recall, F1 = metrics_cal(predicts_lm, tags_train)

        if acc > best_threshold_acc[0]:
            best_threshold_acc[0] = acc
            best_threshold_acc[1] = threshold_sample

        if F1 > best_threshold_f1[0]:
            best_threshold_f1[0] = F1
            best_threshold_f1[1] = threshold_sample

        thresholds_acc.append(acc)
        thresholds_f1.append(F1)

    if metric == 'f1':
        threshold_chosen = best_threshold_f1[1]
    elif metric == 'acc':
        threshold_chosen = best_threshold_acc[1]

    predicts_lm_train = [int(x <= threshold_chosen) for x in loglikelihoods]
    acc, precision, recall, F1 = metrics_cal(predicts_lm_train, tags_train)

    if show_result:
        print(Counter(predicts_lm_train))
        logger.info("average loglikelihood: {} max: {} min: {}".format(np.mean(loglikelihoods), np.max(loglikelihoods), np.min(loglikelihoods)))
        logger.info("threshold: {}".format(threshold_chosen))
        logger.info('Trigram Language Model train' + ": acc: {} precision: {} recall: {} F1: {}".format(
                round(acc, 2), round(precision, 2),
                round(recall, 2), round(F1, 2)))

    if return_likelihood:
        return trigramProb, bigramProb, threshold_chosen, loglikelihoods, V
    else:
        return trigramProb, bigramProb, threshold_chosen, V


def languageModelDev(tokens_tags, trigramProb, bigramProb, threshold_chosen, vocab_size, ngram=3, smoothing=True, k=1,
                      lengthAverage=True, metric='f1', criterion='gridSearch', interval=0.05,
                      show_result=True, return_likelihood=False, note=''):

    tokens_test = list(itertools.chain([' '.join(x[0]).split() for x in tokens_tags]))
    tags_test = list(itertools.chain([int(x[1]) for x in tokens_tags]))
    tokens_trigram, _, _, _ = ngram_model(tokens_test, n=ngram, pad_left=True, pad_right=True,
                                       left_pad_symbol="<SOS>", right_pad_symbol="<EOS>")

    thresholds_acc = []
    thresholds_f1 = []
    best_threshold_acc = [0, 0]
    best_threshold_f1 = [0, 0]

    loglikelihoods = []
    for token_trigrams in tokens_trigram:
        token_score = 0
        for length, trigram in enumerate(token_trigrams):
            # token_score += np.log((trigramProb.get(trigram, 0) + smoothing) / (
            #         bigramProb.get(' '.join(trigram.split()[:-1]), 0) + bigramProb['vol']))
            token_score += np.log((trigramProb.get(trigram, 0) + k*smoothing) / (
                    bigramProb.get(' '.join(trigram.split()[:-1]), 0) + k*vocab_size))
        if lengthAverage:
            loglikelihoods.append(token_score / (length + 1))
        else:
            loglikelihoods.append(token_score)


    # use the threshold from training set to see the 
    predicts_lm_test = [int(x <= threshold_chosen) for x in loglikelihoods]
    acc, precision, recall, F1 = metrics_cal(predicts_lm_test, tags_test)
    if show_result:
        logger.info("Training Set Threshold")
        logger.info("Dev: {}".format(str(Counter(predicts_lm_test))))
        logger.info("Trigram Language Model {}: acc: {} precision: {} recall: {} F1: {}".format(
                note, round(acc, 2), round(precision, 2),
                round(recall, 2), round(F1, 2)))

    # choose the threshold that will have the best F1
    for threshold_sample in np.arange(np.min(loglikelihoods), np.max(loglikelihoods), interval):
        predicts_lm = [int(x <= threshold_sample) for x in loglikelihoods]
        #         print(Counter(predicts_lm))
        acc, precision, recall, F1 = metrics_cal(predicts_lm, tags_test)
        #         print("acc: ", acc, " F1: ", F1)

        if acc > best_threshold_acc[0]:
            best_threshold_acc[0] = acc
            best_threshold_acc[1] = threshold_sample

        if F1 > best_threshold_f1[0]:
            best_threshold_f1[0] = F1
            best_threshold_f1[1] = threshold_sample

        thresholds_acc.append(acc)
        thresholds_f1.append(F1)

    if metric == 'f1':
        dev_threshold = best_threshold_f1[1]
    elif metric == 'acc':
        dev_threshold = best_threshold_acc[1]

    predicts_lm_dev = [int(x <= dev_threshold) for x in loglikelihoods]
    acc, precision, recall, F1 = metrics_cal(predicts_lm_dev, tags_test)

    if show_result:
        print("Dev Set Threshold")
        logger.info("Dev Set Threshold")
        logger.info(Counter(predicts_lm_test))
        logger.info("Trigram Language Model {}: acc: {} precision: {} recall: {} F1: {}".format(
                note, round(acc, 2), round(precision, 2),
                round(recall, 2), round(F1, 2)))

    if return_likelihood:
        return acc, precision, recall, F1, dev_threshold, loglikelihoods
    else:
        return acc, precision, recall, F1, dev_threshold

def languageModelEval(tokens_tags, trigramProb, bigramProb, threshold_chosen, vocab_size, ngram=3, smoothing=True, k=1,
                      lengthAverage=True, metric='f1', criterion='gridSearch', interval=0.05,
                      show_result=True, return_likelihood=False, note=''):

    tokens_test = list(itertools.chain([' '.join(x[0]).split() for x in tokens_tags]))
    tags_test = list(itertools.chain([int(x[1]) for x in tokens_tags]))
    tokens_trigram, _, _, _ = ngram_model(tokens_test, n=ngram, pad_left=True, pad_right=True,
                                       left_pad_symbol="<SOS>", right_pad_symbol="<EOS>")

    loglikelihoods = []

    for token_trigrams in tokens_trigram:
        token_score = 0
        for length, trigram in enumerate(token_trigrams):
            # token_score += np.log((trigramProb.get(trigram, 0) + smoothing) / (
            #         bigramProb.get(' '.join(trigram.split()[:-1]), 0) + bigramProb['vol']))
            token_score += np.log((trigramProb.get(trigram, 0) + k*smoothing) / (
                    bigramProb.get(' '.join(trigram.split()[:-1]), 0) + k*vocab_size))
        if lengthAverage:
            loglikelihoods.append(token_score / (length + 1))
        else:
            loglikelihoods.append(token_score)

    predicts_lm_test = [int(x <= threshold_chosen) for x in loglikelihoods]
    acc, precision, recall, F1 = metrics_cal(predicts_lm_test, tags_test)
    # pdb.set_trace()

    if show_result:
        logger.info(Counter(predicts_lm_test))
        logger.info("Trigram Language Model {}: acc: {} precision: {} recall: {} F1: {}".format(
                note, round(acc, 2), round(precision, 2),
                round(recall, 2), round(F1, 2)))

    if return_likelihood:
        return acc, precision, recall, F1, loglikelihoods
    else:
        return acc, precision, recall, F1

if __name__ == "__main__":
    from collections import defaultdict
    import pickle
    import random
    import string
    from config import *
    from Useful_Function import character_preprocessing
    from OnlineSpellingServer import build_vocab, build_minion_group
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lang", "--lang", type=str, help="language to  train", default="English",
                        choices=["English", "eng_TOEFL", "Finnish", "Italian", "Ainu", "Griko", "Russian", "Spanish", "Turkish"])
    parser.add_argument("-corpus", "--corpus", type=str, help="which corpus to use", default='table',
                        choices=['table', 'corpus', 'corpus_log', 'normal_highrate', 'normal_normalrate', "confusion_set"])
    parser.add_argument("--note", type=str, help="note to mention in the result and model", default='')
    parser.add_argument("--lengthAvg", type=lambda x: (str(x).lower() == 'true'), help="whether to take average of log probability", default=True)
    parser.add_argument("--smoothing", type=lambda x: (str(x).lower() == 'true'), help="whether to use smoothing in ngram model", default=True)
    parser.add_argument("--add_k", type=float, help="add-k smoothing", default=1)
    parser.add_argument("--save_model", type=lambda x: (str(x).lower() == 'true'), help="whether to save the model parameters", default=False)
    parser.add_argument("--save_result", type=lambda x: (str(x).lower() == 'true'), help="whether to save the prediction result", default=False)
    args = parser.parse_args()
    trained_words = []
    lang_folder = args.lang
    smoothing = args.smoothing
    k = args.add_k # add-k smoothing in langauge model
    length_avg = args.lengthAvg
    save_result = args.save_result
    save_model = args.save_model
    corpus = args.corpus # table/confusion_set

    if lang_folder == 'eng_TOEFL':
        langcode = 'en'
        lang_path_addition = ''
    elif lang_folder == 'English':
        langcode = lang2code[lang_folder]
        lang_path_addition = '/USCities'
    else:
        langcode = lang2code[lang_folder]
        lang_path_addition = ''
        
    if corpus in ['table', 'normal_highrate', 'normal_normalrate', 'confusion_set']:
        corpus_file = ''
    else:
        corpus_file = '_' + corpus
    
    if lang_folder == 'Ainu':
        nums = [50] + [x for x in np.arange(100, 2000, 100)]
    elif lang_folder == 'Griko':
        nums = [50] + [x for x in np.arange(100, 2000, 100)]
    elif corpus == 'corpus_log':
        nums = [50] + [x for x in np.arange(100, 3000, 100)]
    else:
        nums = [50] + [x for x in np.arange(100, 3100, 100)]
    summary = defaultdict(list)

    for train_volume in nums:  
        if corpus == 'normal_highrate':
            with open('./files/{}/city{}/highrate/Eng_highrate_label_corpus_{}.txt'.format(lang_folder, lang_path_addition,
                                                                                                        train_volume), 'r') as fp:
                train = fp.readlines()
                train = [x.strip('\n').split() for x in train]
                train = [x for x in train if '\u200b' or '°' not in x[0] and len(x) > 0]
            fp.close()

        elif corpus == 'normal_normalrate':
            with open('./files/{}/city{}/normalrate/Eng_normalrate_label_corpus_{}.txt'.format(lang_folder, lang_path_addition,
                                                                                                        train_volume), 'r') as fp:
                train = fp.readlines()
                train = [x.strip('\n').split() for x in train]
                train = [x for x in train if '\u200b' or '°' not in x[0] and len(x) > 0]
            fp.close()

        else:
            with open('./files/{}/city{}/high_frequent/{}_most_frequent_{}_table{}.txt'.format(lang_folder, lang_path_addition,
                                                                                        langcode, train_volume, corpus_file), 'r') as fp:
                train = fp.readlines()
                train = [x.strip('\n').split() for x in train]
                train = [x for x in train if '\u200b' or '°' not in x[0] and len(x) > 0]
            fp.close()

        lang_folder_older = lang_folder
        lang_folder = 'eng_TOEFL'
        lang_path_addition_older = lang_path_addition
        lang_path_addition = ''
        # dev set (shares 27% vocabulary with training set)
        with open('./files/{}/city{}/high_frequent/{}_most_frequent_dev_200.txt'.format(lang_folder, lang_path_addition, langcode), 'r') as fp:
            dev = fp.readlines()
            dev = [x.strip('\n').split() for x in dev]
            train = [x for x in train if '\u200b' or '°' not in x[0] and len(x) > 0]
        fp.close()

        # test set
        with open('./files/{}/city{}/high_frequent/{}_most_frequent_1000_in_whole_oov_200.txt'.format(lang_folder, lang_path_addition, langcode), 'r') as fp:
            test = fp.readlines()
            test = [x.strip('\n').split() for x in test]
            test = [x for x in test if '\u200b' or '°' not in x[0] and len(x) > 0]
        fp.close()

        # test set (oov_1 and oov_2 shares 10% in vocabulary)
        with open('./files/{}/city{}/high_frequent/{}_most_frequent_1000_in_whole_oov_200_2.txt'.format(lang_folder, lang_path_addition, langcode), 'r') as fp:
            test2 = fp.readlines()
            test2 = [x.strip('\n').split() for x in test2]
            test2 = [x for x in test2 if '\u200b' or '°' not in x[0] and len(x) > 0]
        fp.close()

        # pure correct set
        try:
            with open("./files/{}/city{}/high_frequent/{}_most_frequent_200_in_whole_oov_200_pure_correct.txt".format(lang_folder, lang_path_addition, langcode), "r") as fp:
                pure_correct = fp.readlines()
                pure_correct = [x.strip('\n').split() for x in pure_correct]
                pure_correct = [x for x in pure_correct if '\u200b' or '°' not in x[0] and len(x) > 0]
            fp.close()

            pure_correct_words = [character_preprocessing(x[0]) for x in pure_correct]
            pure_correct_tags = [int(x[1]) for x in pure_correct]
        except:
            pure_correct_words = []
            pure_correct_tags = []    

        lang_folder = lang_folder_older
        lang_path_addition = lang_path_addition_older

        train_words_ = [character_preprocessing(x[0]) for x in train]
        train_words = [x for x in train_words_ if x not in trained_words]
        train_tags = [0 for _ in range(len(train_words))]

        if corpus == 'confusion_set':
            train_seed = 0
            minion_group_size = 15
            minion_group_neg_ratio = 0.5
            misspell_mode = 'byrule'
            max_token_length = 16
            random.seed(train_seed)
            batch_hard_pad = []
            batch_hard_tokens = []
            vocab2index, index2vocab = build_vocab(langcode)
            token_lower = [x.lower().strip(string.punctuation) for x in train_words]
            token_lower = [x for x in token_lower if len(x) > 0]
            for idx, x in enumerate(token_lower):
                minion_group, minion_group_tokens = build_minion_group(langcode, vocab2index, x, minion_group_size,
                                                                        minion_group_neg_ratio, mode=misspell_mode,
                                                                        max_token_length=max_token_length)
                # batch_hard_pad.append(minion_group)
                batch_hard_tokens += minion_group_tokens

            train_words = [x[0] for x in batch_hard_tokens]
            train_tags = [x[1] for x in batch_hard_tokens]

        dev_words = [character_preprocessing(x[0]) for x in dev]
        dev_tags = [int(x[1]) for x in dev]

        test_words = [character_preprocessing(x[0]) for x in test]
        test_tags = [int(x[1]) for x in test]

        test2_words = [character_preprocessing(x[0]) for x in test2]
        test2_tags = [int(x[1]) for x in test2]

        for dev_ in dev:
            if dev_[1] == '0':
                position = '0'
            else:
                for index, (w1, w2) in enumerate(zip(dev_[0], dev_[2])):
                    if w1!=w2:
                        position = index
                        break
            dev_.insert(3, str(position))
            
        dev_pos_tags = [int(x[3]) for x in dev]
        logger.info("----------------------- [lang={}][vol={}] Train -------------------------------".format(lang_folder, train_volume))
        trigramProb, bigramProb, threshold, vocab_size = languageModelTrain([[x, y] for x, y in zip(train_words, train_tags)], smoothing=smoothing, k=k, lengthAverage=length_avg)
        logger.info("----------------------- [lang={}][vol={}] Dev -------------------------------".format(lang_folder, train_volume))
        metric_lm_dev = languageModelEval(dev, trigramProb, bigramProb, threshold, vocab_size=vocab_size, ngram=3, smoothing=smoothing, k=k, lengthAverage=length_avg)
        logger.info("----------------------- [lang={}][vol={}] OOV1 -------------------------------".format(lang_folder, train_volume))
        metric_lm_test = languageModelEval(test, trigramProb, bigramProb, threshold, vocab_size=vocab_size, ngram=3, smoothing=smoothing, k=k, lengthAverage=length_avg)
        logger.info("----------------------- [lang={}][vol={}] OOV2 -------------------------------".format(lang_folder, train_volume))
        metric_lm_test2 = languageModelEval(test2, trigramProb, bigramProb, threshold, vocab_size=vocab_size, ngram=3, smoothing=smoothing, k=k, lengthAverage=length_avg)

        summary['corpus_size'].append(train_volume)
        summary['threshold'].append(threshold)
        summary['dev_metrics'].append(metric_lm_dev)
        summary['test_metrics'].append(metric_lm_test)
        summary['test2_metrics'].append(metric_lm_test2)

        if len(pure_correct):
            logger.info("----------------------- [lang={}][vol={}] Pure Correct -------------------------------".format(lang_folder, train_volume))
            metric_lm_correct = languageModelEval(pure_correct, trigramProb, bigramProb, threshold, vocab_size=vocab_size, ngram=3, smoothing=smoothing, k=k, lengthAverage=length_avg)
            summary['pure_correct_metrics'].append(metric_lm_correct)

        print('Finish {}'.format(train_volume))

    note = "-{}".format(args.note) if len(args.note) else ''
    if save_result:
        with open('./report/LanguageModel/{}-{}-triLM-corpus_{}-lengthAvg_{}-k_{}{}.pkl'.format(lang_folder, train_volume, corpus, length_avg, k, note), 'wb') as fp:
            pickle.dump(summary, fp)
        fp.close()

    if save_model:
        model = {'trigram': trigramProb, 'bigram': bigramProb, 'ngram': 3, 'threshold':threshold, 'smoothing':smoothing, 'k':k, 'lengthAvg': length_avg, 'vocab_size': vocab_size}
        with open('./report/LanguageModel/model/{}-{}-triLM-corpus_{}-lengthAvg_{}-k_{}{}model.pkl'.format(lang_folder, train_volume, corpus, length_avg, k, note), 'wb') as fp:
            pickle.dump(model, fp)
        fp.close()
    # pdb.set_trace()

