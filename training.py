import os
import string
import time
import random
import pandas as pd
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from RuPreprocess import RNN, evaluate,\
                        get_distance, \
                        load_model, save_model, \
                        tuning_model
from config import *
from OnlineSpellingServer import get_model, \
                                 build_vocab, get_optimizer, \
                                 get_loss_criterion, build_minion_group, \
                                 token2idxs
from Useful_Function import softmax, character_preprocessing, get_timestamp
from collections import defaultdict, Counter
from sklearn.metrics import precision_recall_fscore_support
from evaluation import metrics_cal
from tfidf_model import tfidf_feature
from language_model import languageModelTrain, languageModelEval
# from universial_config import args
import pdb

import logging

current_lang = 'en'
lang = 'English'

minion_group_neg_ratio = 0.5
use_batch = True
decisions = {0: 'correct', 1: 'misspelled'}
decisions2Tag = {'correct':0, 'misspelled':1}

# hidden_size = 30
# embedding_size = 50
# n_layers = 3
# dropout = 0.1
weights = [1, 0]
tune_epoch = 10
max_token_length = 16
loss_criterion = "CrossEntropy"
bidirection = False
USE_CUDA = False
best_model = False
best_metric = 0
pattern = "random_zero_shot"
activation = "lstm"
decisions = {0: 'correct', 1: 'misspelled'}
best_result_by_lang = defaultdict(dict)
best_result = {'acc': {}, 'f1': {}}
best_model_parameter = {'model': {}, 'optimizer': {}}
word_similarity = defaultdict(dict)
Type = torch.LongTensor
early_stop = 2 #2
eval_every_iter = 40
random_seed = 0

best_acc = 0.6
best_f1 = 0

def evaluation(tokens, vocab2index, activation, model, Type, args):
    # global vocab2index, activation, model, Type

    evaluate_sample = [token2idxs(x, vocab2index, mode='padding') for x in tokens]

    sample_scores, sample_predicts, hiddens = evaluate_(evaluate_sample, vocab2index, args.model, model,
                                                        activation, Type=Type, return_hidden=True)
    sample_scores_show = sample_scores.detach().numpy()
    sample_prob = softmax(sample_scores_show, axis=1)
    if len(tokens) > 1:
        try:
            result_model = [decisions[x] for x in sample_predicts.tolist()]
        except TypeError:
            pdb.set_trace()
    else:
        result_model = decisions[sample_predicts.tolist()[0]]
    return sample_prob, result_model


def evaluate_(data, word2index, model_name, model, activation, Type=torch.LongTensor, return_hidden=False):
    # if len(data) < 16:
    #     data += [word2index['<PAD>']] * (16 - len(data))
    dev_tensor = torch.tensor(data).type(Type)
    if type(data) == list:
        length = len(data)
    else:
        length = 1
    model.eval()
    if model_name == 'cnn':
        encoder_outputs = 0
        scores = model(dev_tensor, length)  # batch * 1 * ntags
        predicts = scores.argmax(dim=1).cpu().numpy()
    if model_name == 'rnn':
        encoder_outputs, encoder_hidden = model(dev_tensor, length, padded=False, hidden=None)
        encoder_last_outputs = encoder_outputs[:, -1, :]
        scores = model.projection(encoder_last_outputs)
        predicts = scores.argmax(dim=1).cpu().numpy()

    if return_hidden:
        return scores, predicts, encoder_outputs
    else:
        return scores, predicts


def evaluate_wrapper(test_words, labels, vocab2index, model, Type, args, activation='lstm', title="Evaluation", print_result=True, return_prob=False, show_dist=False,
                     sklearn_mode='macro'):
    pred_probs, predicts = evaluation(test_words, vocab2index, activation, model, Type, args)
    predicts_tags = [decisions2Tag[x] for x in predicts]
    if show_dist:
        print("labels: ", Counter(labels))
        print("predicts: ", Counter(predicts_tags))
        logger.info("labels: " + str(Counter(labels)))
        logger.info("predicts: " + str(Counter(predicts_tags)))

    acc, precision, recall, F1 = metrics_cal(predicts_tags, labels, sklearn_mode=sklearn_mode)

    if print_result:
        print(title + ": acc: {} (best {}) precision: {} recall: {} F1: {} (best {})".format(
            round(acc, 2), round(best_acc, 2), round(precision, 2),
            round(recall, 2), round(F1, 2), round(best_f1, 2)))
        logger.info(title + ": acc: {} (best {}) precision: {} recall: {} F1: {} (best {})".format(
            round(acc, 2), round(best_acc, 2), round(precision, 2),
            round(recall, 2), round(F1, 2), round(best_f1, 2)))

    if return_prob:
        return acc, precision, recall, F1, pred_probs
    else:
        return acc, precision, recall, F1


def save_ckpt(state, is_best, result, metric='F1', filename='checkpoint'):
    torch.save(state, filename+'_model.pth.tar')
    # torch.save(state['model'], filename+'_model.pth.tar')
    # torch.save(state['optimizer'], filename + '_optimizer.pth.tar')

    # copy the file to best model
    # if is_best:
    #     shutil.copyfile(filename+'_model.pth.tar', '{}_model_best_{}_{}.pth.tar'.format(state['lang'], metric, result))
    #     shutil.copyfile(filename + '_optimizer.pth.tar', '{}_model_best_{}_{}_optimizer.pth.tar'.format(state['lang'], metric, result))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-lang", "--lang", type=str, help="lang", default='English')
    parser.add_argument("-model", "--model", type=str, help="model", default='cnn', choices=['rnn', 'birnn', 'triLM', 'cnn'])
    parser.add_argument("-corpus", "--corpus", type=str, help="which corpus to use", default='table',
                        choices=['table', 'corpus', 'corpus_log', 'normal_highrate', 'normal_normalrate'])
    parser.add_argument("-num", "--num", type=int, help="training volume", default=50)
    parser.add_argument("-layers", "--n_layers", type=int, help="layer of nn", default=3)
    parser.add_argument("-hidden", "--hidden_size", type=int, help="hidden size of cell", default=30)
    parser.add_argument("-embedding", "--embedding_size", type=int, help="embedding size of characters", default=30)
    parser.add_argument("-drop", "--dropout", type=float, help="drop out rate", default=0.1)
    parser.add_argument("-lr", "--lr", type=float, help="learning rate", default=0.00001)
    parser.add_argument("-lr_decay", "--lr_decay", type=float, help="decay of learning rate", default=1)
    parser.add_argument("-num_seeds", "--num_seeds", type=int, help="number of random seeds", default=5)
    parser.add_argument("-ckpt_path", "--ckpt_path", type=str, help="directory name of ckpts", default='date')
    parser.add_argument("-mode", "--mode", type=str, help="training mode: incremental/individual", default='individual')
    parser.add_argument("-stop_mode", "--stop_mode", type=str, help="stop mode: oneEpoch/early_stop", default='oneEpoch')
    parser.add_argument("-save_mode", "--save_mode", type=str, help="when to save model",
                        default='bySeed')
    parser.add_argument("-minion_group", "--minion_group", type=int, help="size of minion group for each word in the vocabulary",
                        default=15)
    parser.add_argument("-batch_size", "--batch_size", type=int, help="number of samples in the batch", default=15)
    parser.add_argument("-misspell_mode", "--misspell_mode", type=str, help="mode of misspelling generation", default='byrule')
    parser.add_argument("-fresh_summary", "--fresh_summary", default=True, help="whether to write summary in a fresh file", type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("-use_cuda", "--use_cuda", default=False, help="whether to use cuda", type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("-skip", "--skip", default=False, help="skip certain words in the training",
                        type=lambda x: (str(x).lower() == 'true'))
    # learning_rate = 0.00001  # 0.001
    # global args
    args = parser.parse_args()
    dropout = args.dropout
    pure_correct_exception = ['Ainu', 'Griko', 'Hinglish']

    lang_folder = args.lang

    if args.corpus in ['table', 'normal_highrate', 'normal_normalrate']:
        corpus_file = ''
    else:
        corpus_file = '_' + args.corpus

    if lang_folder == 'eng_TOEFL':
        langcode = 'en'
        lang_path_addition = ''
    elif lang_folder == 'English':
        langcode = lang2code.get(lang_folder, lang_folder)
        lang_path_addition = '/USCities'
    else:
        langcode = lang2code.get(lang_folder, lang_folder)
        lang_path_addition = ''

    langCode = langcode
    current_lang = langCode
    vocab2index, index2vocab = build_vocab(langCode)

    logging.basicConfig(level=logging.INFO,
                        filename='./log/{}-random_zero_shot.txt'.format(langcode),
                        filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    random.seed(0)
    random_seed_set = random.sample([x for x in range(20)], args.num_seeds)

    # for real misspellings
    # nums_set = [x for x in np.arange(100, 3100, 100)]

    # for testing
    nums_set = [x for x in np.arange(100, 3100, 100)]

    if langcode == 'griko':
        nums_set = [x for x in np.arange(100, 3000, 100)]

    elif langcode == 'ainu':
        nums_set = [x for x in np.arange(100, 2000, 100)]

    # nums_set = [50] + [x for x in np.arange(100, 5100, 100)]
    # nums_set = [50] + [x for x in np.arange(100, 2500, 100)]

    reload_model = False
    best_model_metric = 'F1'
    logger.info("------------------------------------- Setting --------------------------------------------------------")
    logger.info("Experiment for {} [mode={}] from seeds {} to {}".format(args.lang, args.mode, nums_set[0], nums_set[-1]))
    logger.info("[lr={}]".format(args.lr))
    logger.info("[batch={}]".format(args.batch_size))
    logger.info("[minion_group_size={}]".format(args.minion_group))
    logger.info("[misspell_mode={}]".format(args.misspell_mode))
    logger.info("[corpus={}]".format(args.corpus))
    logger.info("------------------------------------------------------------------------------------------------------")

    # Ainu
    # nums_set = [x for x in np.arange(100, 1000, 100)]

    minion_group_size = args.minion_group
    batch_size = args.batch_size
    learning_rate = args.lr

    cuda = args.use_cuda

    if args.ckpt_path == 'date':
        date = 'date[{}]'.format(get_timestamp().replace(":", "#"))
        ckpt_path = lang + '-{}'.format(date)

    ckpt_dir = './model/{}/{}'.format(lang, ckpt_path) if ckpt_path != '' else './model/{}'.format(lang)
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)

    if not args.fresh_summary:
        summary_file = './report/IncrementalTest/{}_summary.csv'.format(langcode)
        if os.path.exists(summary_file):
            data_result = pd.read_csv(summary_file)
            data_result = data_result.loc[:, ~data_result.columns.str.match('Unnamed')]
        else:
            data_result = pd.DataFrame(columns=summary_columns)
    else:
        summary_file = './report/IncrementalTest/{}-{}-summary.csv'.format(langcode, date)
        data_result = pd.DataFrame(columns=summary_columns)

    if lang_folder == 'Hinglish':
        folder = 'English'
        path_addition = '/USCities'
        code = 'en'
        # test set
        with open('./files/{}/city{}/high_frequent/{}_most_frequent_1000_in_whole_oov_200.txt'.format(folder,
                                                                                                    path_addition,
                                                                                                    code), 'r') as fp:
            test = fp.readlines()
            test = [x.strip('\n').split() for x in test]
            test = [x for x in test if '\u200b' or '°' not in x[0] and len(x) > 0]
        fp.close()

        # test set (oov_1 and oov_2 shares 10% in vocabulary)
        with open('./files/{}/city{}/high_frequent/{}_most_frequent_1000_in_whole_oov_200_2.txt'.format(folder,
                                                                                                        path_addition,
                                                                                                        code), 'r') as fp:
            test2 = fp.readlines()
            test2 = [x.strip('\n').split() for x in test2]
            test2 = [x for x in test2 if '\u200b' or '°' not in x[0] and len(x) > 0]
        fp.close()
    else:
        # lang_folder_older = lang_folder
        # lang_folder = 'eng_TOEFL'
        # lang_path_addition_older = lang_path_addition
        # lang_path_addition = ''

        # test set
        with open('./files/{}/city{}/high_frequent/{}_most_frequent_1000_in_whole_oov_200.txt'.format(lang_folder,
                                                                                                    lang_path_addition,
                                                                                                    langcode), 'r') as fp:
            test = fp.readlines()
            test = [x.strip('\n').split() for x in test]
            test = [x for x in test if '\u200b' or '°' not in x[0] and len(x) > 0]
        fp.close()

        # test set (oov_1 and oov_2 shares 10% in vocabulary)
        with open('./files/{}/city{}/high_frequent/{}_most_frequent_1000_in_whole_oov_200_2.txt'.format(lang_folder,
                                                                                                        lang_path_addition,
                                                                                                        langcode),
                'r') as fp:
            test2 = fp.readlines()
            test2 = [x.strip('\n').split() for x in test2]
            test2 = [x for x in test2 if '\u200b' or '°' not in x[0] and len(x) > 0]
        fp.close()

        # lang_folder = lang_folder_older
        # lang_path_addition = lang_path_addition_older

    # dev set (shares 27% vocabulary with training set)
    try:
        if lang_folder == 'Hinglish':
            with open('./files/{}/city{}/high_frequent/{}_most_frequent_dev_200.txt'.format(folder, path_addition,
                                                                                            code), 'r') as fp:
                dev = fp.readlines()
                dev = [x.strip('\n').split() for x in dev]
                dev = [x for x in dev if '\u200b' or '°' not in x[0] and len(x) > 0]
            fp.close()

            dev_words = [character_preprocessing(x[0]) for x in dev]
            dev_tags = [int(x[1]) for x in dev]

            for dev_ in dev:
                if dev_[1] == '0':
                    position = '0'
                else:
                    for index, (w1, w2) in enumerate(zip(dev_[0], dev_[2])):
                        if w1 != w2:
                            position = index
                            break
                dev_.insert(3, str(position))
            dev_pos_tags = [int(x[3]) for x in dev]
        else:
            with open('./files/{}/city{}/high_frequent/{}_most_frequent_dev_200.txt'.format(lang_folder, lang_path_addition,
                                                                                            langcode), 'r') as fp:
                dev = fp.readlines()
                dev = [x.strip('\n').split() for x in dev]
                dev = [x for x in dev if '\u200b' or '°' not in x[0] and len(x) > 0]
            fp.close()

            dev_words = [character_preprocessing(x[0]) for x in dev]
            dev_tags = [int(x[1]) for x in dev]

            for dev_ in dev:
                if dev_[1] == '0':
                    position = '0'
                else:
                    for index, (w1, w2) in enumerate(zip(dev_[0], dev_[2])):
                        if w1 != w2:
                            position = index
                            break
                dev_.insert(3, str(position))
            dev_pos_tags = [int(x[3]) for x in dev]

    except:
        dev = []
        dev_words = []
        dev_tags = []

    test_words = [character_preprocessing(x[0]) for x in test]
    test_tags = [int(x[1]) for x in test]

    test2_words = [character_preprocessing(x[0]) for x in test2]
    test2_tags = [int(x[1]) for x in test2]

    logger.info("Dev: " + str(len(dev_words)) + ' misspelling: ' + str(len([x for x in dev if dev[1] == '1'])))

    if lang_folder not in pure_correct_exception:
        # pure correct set
        with open("./files/{}/city{}/high_frequent/{}_most_frequent_200_in_whole_oov_200_pure_correct.txt".format(
                lang_folder, lang_path_addition, langcode), "r") as fp:
            pure_correct = fp.readlines()
            pure_correct = [x.strip('\n').split() for x in pure_correct]
            pure_correct = [x for x in pure_correct if '\u200b' or '°' not in x[0] and len(x) > 0]
        fp.close()
        pure_correct_words = [character_preprocessing(x[0]) for x in pure_correct]
        pure_correct_tags = [int(x[1]) for x in pure_correct]


    # choose a random seed
    for seed_index, train_seed in enumerate(random_seed_set):
        # if not reload_model:
        model = get_model(args.model, len(index2vocab), args)
        optimizer = get_optimizer(model, 'adam')
        criterion = get_loss_criterion(loss_criterion)

        # store the words that already in the training set (keep empty for individual mode)
        if args.skip:
            trained_words = ['center']
        else:
            trained_words = []
        skip_words = trained_words.copy()

            # if args.mode == 'incremental':
            #     print('incremental training')
            #     reload_model = True

        # choose number of seed words
        for num in nums_set:
            # otherwise only save the best model
            best_model = True if args.save_mode == 'bySeed' else False
            stats_summary = defaultdict(list)
            train_volume = num
            mode = args.mode  # or incremental

            best_model_ckpt = {}

            if args.corpus == 'normal_highrate':
                with open('./files/{}/city{}/highrate/Eng_highrate_label_corpus_{}.txt'.format(lang_folder, lang_path_addition,
                                                                                                             train_volume), 'r') as fp:
                    train = fp.readlines()
                    train = [x.strip('\n').split() for x in train]
                    train = [x for x in train if '\u200b' or '°' not in x[0] and len(x) > 0]
                fp.close()

            elif args.corpus == 'normal_normalrate':
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

            train_words_ = [character_preprocessing(x[0]) for x in train]
            train_words = [x for x in train_words_ if x not in trained_words]
            train_tags = [0 for _ in range(len(train_words))]

    ####################################################################################################

            random.seed(train_seed)
            batch_hard_pad = []
            batch_hard_tokens = []
            token_lower = [x.lower().strip(string.punctuation) for x in train_words]
            token_lower = [x for x in token_lower if len(x) > 0]

            if len(token_lower):

                fail_counter = 0

                for idx, x in enumerate(token_lower):
                    minion_group, minion_group_tokens = build_minion_group(current_lang, vocab2index, x, minion_group_size,
                                                                           minion_group_neg_ratio, mode=args.misspell_mode,
                                                                           max_token_length=max_token_length)
                    batch_hard_pad.append(minion_group)
                    batch_hard_tokens += minion_group_tokens

                # pdb.set_trace()

                start_time = time.time()
                # Summary and log
                # early stop only after the first epoch?
                summary = {'loss': [], 'accuracy': [], 'dev_acc': [], 'dev_f1': [], 'dev_acc_pos': [], 'trigger': ['None']}
                summary['mode'] = mode
                summary['lang'] = lang_folder
                summary['corpus'] = args.corpus
                summary['langcode'] = langcode
                summary['random_seed'] = train_seed
                summary['volume'] = len(token_lower) + len(trained_words)
                summary['lr_decay'] = args.lr_decay
                summary['mode'] = args.mode
                summary['weight'] = weights
                summary['tune_epoch'] = tune_epoch
                summary['batch_size'] = batch_size
                summary['dropout'] = dropout
                summary['n_layers'] = args.n_layers
                summary['early_stop'] = int(early_stop)
                summary['embedding_size'] = args.embedding_size
                summary['hidden_size'] = args.hidden_size
                summary['lr'] = learning_rate
                summary['minion_group_size'] = minion_group_size
                summary['minion_group_neg_ratio'] = minion_group_neg_ratio
                summary['max_token_length'] = max_token_length
                summary['stop_mode'] = args.stop_mode
                summary['skip'] = args.skip

                if len(token_lower) == 0:
                    pdb.set_trace()

                if len(dev_words):
                    batch_hard_pad, summary = tuning_model(model, optimizer, vocab2index,
                                                           (token_lower, token_lower), 0, batch_hard_pad, args,
                                                           summary, tune_epoch=tune_epoch, Type=Type, criterion=criterion,
                                                           batch=use_batch, batch_size=batch_size, protect_epoch=0,
                                                           weights=weights, eval_every_iter=eval_every_iter,
                                                           hard_sample_stop=False, early_stop=early_stop,
                                                           dev_set=(dev_words, dev_tags, dev_pos_tags), model_save=True,
                                                           lr_decay=args.lr_decay, stop_strategy=args.stop_mode, log=True)

                    if len(summary['dev_acc'][0]):
                        summary['last_dev_acc'] = summary['dev_acc'][0][-1]
                        summary['last_dev_f1'] = summary['dev_f1'][0][-1]
                else:
                    batch_hard_pad, summary = tuning_model(model, optimizer, vocab2index,
                                                           (token_lower, token_lower), 0, batch_hard_pad, args,
                                                           summary, tune_epoch=tune_epoch, Type=Type, criterion=criterion,
                                                           batch=use_batch, batch_size=batch_size, protect_epoch=0,
                                                           weights=weights, eval_every_iter=eval_every_iter,
                                                           hard_sample_stop=False, early_stop=early_stop,
                                                           dev_set=(), model_save=True, lr_decay=args.lr_decay,
                                                           stop_strategy=args.stop_mode, log=True)

                print("[correction] tuning model for {} words in {}s".format(len(token_lower), round(time.time() - start_time, 2)))
                logger.info("[correction] tuning model for {} words in {}s".format(len(token_lower), round(time.time() - start_time, 2)))

                test_summary = evaluate_wrapper(test_words, test_tags, vocab2index, model, Type, args, activation, title='Test', show_dist=True, return_prob=True)
                acc, precision, recall, F1, pred_probs_test = test_summary
                summary['test_acc'] = acc
                summary['test_precision'] = precision
                summary['test_recall'] = recall
                summary['test_f1'] = F1
                summary['test_probs'] = pred_probs_test
                test2_summary = evaluate_wrapper(test2_words, test2_tags, vocab2index, model, Type, args, activation, title='Test2', show_dist=True, return_prob=True)
                acc2, precision2, recall2, F12, pred_probs_test2 = test2_summary
                summary['test2_acc'] = acc2
                summary['test2_precision'] = precision2
                summary['test2_recall'] = recall2
                summary['test2_f1'] = F12
                summary['test2_probs'] = pred_probs_test2

                stats_summary['test_acc'].append(acc)
                stats_summary['test_f1'].append(F1)
                stats_summary['test2_acc'].append(acc2)
                stats_summary['test2_f1'].append(F12)

                if best_model_metric == 'F1':
                    current_metric = (F1 + F12) * 0.5

                elif best_model_metric == 'accuracy':
                    current_metric = (acc + acc2) * 0.5

                if current_metric > best_metric:
                    best_model = True
                    best_metric = current_metric
                    best_model_parameter['model'] = model.state_dict()
                    best_model_parameter['optimizer'] = optimizer.state_dict()
                    best_model_parameter['accuracy'] = acc
                    best_model_parameter['F1'] = F1

                print("current: {} best: {} best_model: {}".format(round(current_metric, 2), round(best_metric, 2), best_model))

                if lang_folder not in pure_correct_exception:
                    pure_correct_summary = evaluate_wrapper(pure_correct_words, pure_correct_tags, vocab2index, model, Type, args, activation, title='Pure correct', show_dist=True, return_prob=True)
                    acc_correct, precision_correct, recall_correct, F1_correct, pred_probs_test_correct = pure_correct_summary
                    summary['correct_acc'] = acc_correct
                    summary['correct_precision'] = precision_correct
                    summary['correct_recall'] = recall_correct
                    summary['correct_f1'] = F1_correct
                    summary['correct_probs'] = pred_probs_test_correct
                    stats_summary['correct_acc'].append(acc_correct)
                    stats_summary['correct_f1'].append(F1_correct)
                    stats_summary['dev_acc'] = summary['dev_acc'][-1]
                    stats_summary['dev_f1'] = summary['dev_f1'][-1]

                summary['random_seed'] = int(summary['random_seed'])
                summary['misspell_mode'] = args.misspell_mode
                if type(summary['volume']) == list:
                    summary['volume'] = summary['volume'][0]
                data_result_tmp = pd.Series(summary)
                try:
                    data_result = data_result.append(data_result_tmp, ignore_index=True)
                except:
                    print("save summary failed")
                    pdb.set_trace()

                split_msg = "----------------------- {} volumn seed {} {}/{} finish ----------------------------------------".format(
                    train_volume, train_seed, seed_index+1, len(random_seed_set))
                print(split_msg)
                logger.info(split_msg)
            #########################################################################

                for key, value in stats_summary.items():
                    if type(value) == list:
                        show_value = np.mean(value)
                        show_std = np.std(value)
                    else:
                        show_value = value
                        show_std = 0

                    logger.info('{}: {}'.format(key, show_value))
                    print(key, show_value, show_std)

                try:
                    # TODO: save to the same file of sparate files
                    data_result.to_csv(summary_file)
                except:
                    pdb.set_trace()

                # save the best model
                if best_model:
                    best_model_parameter['lang'] = langcode
                    filename = ckpt_dir + '/{}-{}-mode_{}-corpus_{}-seed_{}-batch_{}-skip_{}_{}-lr_{}-lr_decay_{}-metric_{}-{}_{}-{}_{}-ckpt'.format(lang, langcode, num, mode,
                                                                       args.corpus, train_seed, batch_size, args.skip, ' '.join(skip_words), learning_rate, args.lr_decay, best_model_metric,
                                                                        'Acc',  round(best_model_parameter['accuracy'], 2), \
                                                                        'F1', round(best_model_parameter['F1'], 2))
                    save_ckpt(best_model_parameter, is_best=best_model, result=best_metric,
                              metric=best_model_metric, filename=filename)
                    print("Ckpt saved at {}".format(filename))
                    logger.info("Ckpt saved at {}".format(filename))


                if mode == 'incremental':
                    trained_words += train_words

        else:
            print("no new words")
            logger.info("no new words")

        







