from flask import Flask
from flask import g, request, jsonify
# from werkzeug import secure_filename
from multiprocessing import Process
import requests, json, random
import logging

import sys
import pdb
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import time
import numpy as np
import random
import pickle
import string
import itertools
from collections import defaultdict, Counter
import time
import nltk
import pandas as pd
from Preprocessing import preprocess, get_batches, summary_saver, dict_saver
from Data_loader import DataLoader
from Useful_Function import softmax, misspelling_generator_wrapper, extract_timestamp
from language_model import languageModelTrain, languageModelEval
from config import *
from universial_config import args

import copy

# Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from RuPreprocess import RNN, evaluate,\
                        get_distance, \
                        load_model, save_model, tuning_model, active_save_model, language_model_save
from Model import CNN

# Timely check
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    TimelyCheck = True
except ImportError:
    print("No apscheduler")
    TimelyCheck = False

app = Flask(__name__)

current_lang = 'en'

hidden_size = 30  # 50
embedding_size = 50
n_layers = 3
dropout = 0.1
learning_rate = 0.00001  # 0.001
minion_group_size = 15
max_token_length = 16
batch_size = 15
weights = [1, 0]
eval_every_iter = 50
early_stop = 2
stop_mode = 'oneEpoch'
lr_decay = 1
tune_epoch = 1
hard_sample_stop = False
loss_criterion = "CrossEntropy"
# RNN Parameters
bidirection = False
USE_CUDA = False
pattern = "random_zero_shot"
activation = "lstm"

# CNN Parameters:
# WIN_SIZE = [3, 4, 5]
WIN_SIZE = [2, 3, 4]
dropout = 0.1
# Define the model
FILTER_SIZE = 100
decisions = {0: 'correct', 1: 'misspelled'}
best_result_by_lang = defaultdict(dict)
best_result = {'acc': {}, 'f1': {}}
best_model_parameter = {'model': {}, 'optimizer': {}}
word_similarity = defaultdict(dict)
Type = torch.LongTensor
model = 'CNN' # RNN/triLM/CNN
criterion = "CrossEntropy"

# Language Model Parameters
global trigramProb, bigramProb, lm_threshold, vocab
trigramProb = {}
bigramProb = {}
lm_threshold = 0
vocab = []
LM_model = {'trigramProb':trigramProb, 'bigramProb': bigramProb, 'train_threshold':lm_threshold}

# ckpt number
global max_ckpt_num
max_ckpt_num = 5

# character-level
global vocab2index, index2vocab
vocab2index = {}
index2vocab = {}

# token-level
global token2index, index2token
token2index = {}
index2token = {}
token_size = 0

KB_correction = defaultdict(str)
KB_size = 0

# Retrieval
top_N_retrieval = 10

# Minion group
minion_group_neg_ratio = 0.5 # 0.7
# Summary and log
summary = {'loss': [], 'accuracy': [], 'dev_acc': [], 'dev_f1': [],
        'dev_acc_pos': [], 'trigger': ['None']}

logging.basicConfig(level=logging.INFO,
                    filename='./log/server.log',
                    filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Metrics to observe and collect
global misspell2correct, correct2misspell, retrievalEffectiveClick, retrievalManualCorrect, retrievalClick, accuracy, precision, recall, F1
misspell2correct, correct2misspell, retrievalEffectiveClick, retrievalManualCorrect, retrievalClick = [0, 0, 0, 0, 0]
accuracy, precision, recall, F1 = [0, 0, 0, 0]

# TODO: main a dev set for each language

def get_model(model_name, vocab_size, args):
    if model_name == 'rnn':
        model = RNN(vocab_size, args.hidden_size, args.embedding_size, n_layers=args.n_layers, dropout=dropout,
                    ntags=2, bidirection=False)

    if model_name == 'birnn':
        model = RNN(vocab_size, args.hidden_size, embedding_size, n_layers=args.n_layers, dropout=dropout,
                    ntags=2, bidirection=True)

    elif model_name == 'triLM':
        model = {
            'trigramProb': {},
            'bigramProb': {},
            'train_threshold':0
        }

    elif model_name == 'cnn':
        ntags = 2
        try:
            model = CNN(vocab_size, vocab_size, FILTER_SIZE, WIN_SIZE, dropout, ntags=2, weight_norm=3, Type=torch.LongTensor, pretrained_embedding=None)
        except:
            pdb.set_trace()
    return model

def build_vocab(lang, default='it'):
    alphabet = langCode2alphabet.get(lang, langCode2alphabet[default])
    index2vocab = {i + 1: alphabet[i] for i in range(len(alphabet))}
    index2vocab[0] = '<PAD>'
    index2vocab[len(index2vocab)] = ' '
    index2vocab[len(index2vocab)] = '-'
    index2vocab[len(index2vocab)] = '_'
    index2vocab[len(index2vocab)] = '–'
    index2vocab[len(index2vocab)] = '<UNK>'
    index2vocab[len(index2vocab)] = '<EOS>'
    index2vocab[len(index2vocab)] = '<SOS>'
    vocab2index= {v: k for k, v in index2vocab.items()}
    print("alphabet: ", vocab2index)
    return vocab2index, index2vocab

def get_optimizer(model, name, lr=0.00001, weight_decay=0):
    if name == 'sgd':
        try:
            if weight_decay > 0:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        except AttributeError:
            optimizier = {'name':'no-optimizer'}

    if name == 'adam':
        try:
            if weight_decay > 0:
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                optimizer = optim.Adam(model.parameters(), lr=lr)
        except AttributeError:
            optimizier = {'name':'no-optimizer'}

    return optimizer

def get_loss_criterion(name):
    if name == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    if name == "mse":
        criterion = nn.MSELoss()
    return criterion

def addToken(token):
    global token2index, index2token
    vol = len(token2index)

    if token2index.get(token, '') == '':
        size = len(token2index)
        token2index[token] = size
        index2token[size] = token
        return True, vol

    else:
        return False, vol+1

def removeToken(token):
    global token2index, index2token

    vol = len(token2index)

    value = token2index.pop(token, None)
    if value is not None:
        index2token.pop(value)

    return vol

def token2idxs(token, vocab2index, max_token_length=16, mode='padding'):
    if mode == 'SEOS':
        return [vocab2index['<SOS>']] + [vocab2index.get(x, vocab2index['<UNK>']) for x in token] + [vocab2index['<EOS>']]
    elif mode == 'origin':
        return [vocab2index.get(x, vocab2index['<UNK>']) for x in token]
    elif mode == 'padding':
        raw_idx = [vocab2index['<SOS>']] + [vocab2index.get(x, vocab2index['<UNK>']) for x in token] + [vocab2index['<EOS>']]
        if len(raw_idx) < max_token_length:
            raw_idx += [vocab2index['<PAD>'] for _ in range(max_token_length - len(raw_idx))]
        elif len(raw_idx) > max_token_length:
            raw_idx = raw_idx[:max_token_length - 1]
            raw_idx += [vocab2index['<EOS>']]
        return raw_idx

# misspelling - word - distance is not reasonable because the form of misspelling is unpredicted
def build_word_similarity(KB_correction, word_similarity):
    for spelling, correct_form in KB_correction.items():
        distance = get_distance(spelling, correct_form)
        if distance > 0:
            if word_similarity.get(spelling, '') == '':
                word_similarity[spelling] = defaultdict(set)
            word_similarity[spelling][distance].add(correct_form)

def KB_V_update():
    print("Updating Knowledge base ...")
    # Update Knowledge-base for incremental knowledge
    global current_lang, KB_correction, KB_size, token2index, index2token, token_size
    if current_lang != "":
        if len(KB_correction) > KB_size:
            KB_size = len(KB_correction)
            with open('./KnowledgeBase/KnowledgeBase/{}/kb.pkl'.format(current_lang), 'wb') as fp:
                pickle.dump(KB_correction, fp, protocol=pickle.HIGHEST_PROTOCOL)

            logging.info("{} knowledge updated to size {}".format(current_lang, KB_size))

        # Update vocabulary
        # print("vocab size: {} current size {}".format(len(token2index), token_size))
        if len(token2index) > token_size:
            token_size = len(token2index)
            with open('./KnowledgeBase/Vocabulary/{}/token2index.pkl'.format(current_lang), 'wb') as fp:
                pickle.dump(token2index, fp, protocol=pickle.HIGHEST_PROTOCOL)

            with open('./KnowledgeBase/Vocabulary/{}/index2token.pkl'.format(current_lang), 'wb') as fp:
                pickle.dump(index2token, fp, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info("{} vocab updated to size {}".format(current_lang, token_size))

        logger.info("{} knowledge updated".format(current_lang))
    print("Knowledge base updated")
    metric_update()


def metric_update():
    global misspell2correct, correct2misspell, retrievalEffectiveClick, retrievalManualCorrect, retrievalClick
    logger.info("[summary] misspell2correct: {} correct2misspell: {} retrievalEffectiveClick: {} retrievalManualCorrect: {} retrievalClick: {}".format(
        misspell2correct, correct2misspell, retrievalEffectiveClick, retrievalManualCorrect, retrievalClick
    ))
    print("Metrics collection updated")


def retrieval(token, top_N_retrieval, KB_correction, max_distance=5, trust_origin=True):
    candidate_pool = []

    distances = defaultdict(list)
    volume = 0
    max_distance_real = 0
    min_distance = 0 + int(not trust_origin)

    for word in KB_correction.values():
        if len(word):
            distance = get_distance(word, token)

            # distance > 0 means doesn't treat the word itself to be correct
            if distance <= max_distance and distance >= min_distance:
                distances[distance].append(word)
                volume += 1
                max_distance_real = max(max_distance_real, distance)
            
            if volume > 4 * top_N_retrieval:
                break

    for distance_select in range(min_distance, max_distance_real+1):
        if len(distances[distance_select]) > 0:
            for word in distances[distance_select]:
                candidate_pool.append(word)
                if len(candidate_pool) == top_N_retrieval:
                    break

    return candidate_pool

def token_preprocessing(x):
    x = x.replace('‘', "'").replace('“', '"').replace('’', "'").replace('”', '"')
    recover_actions = []
    if x[0] == "'":
        recover_actions.append("front_abbre")
    if x[-1] == "'":
        recover_actions.append("end_abbre")

    x = x.strip(string.punctuation)

    if len(recover_actions):
        for recover_action in recover_actions:
            if recover_action == 'front_abbre':
                x = "'"+x
            elif recover_action == 'end_abbre':
                x += "'"

    return x

def build_minion_group(lang, vocab2index, token, size, minion_group_neg_ratio, times=1, mode='auto', adding_pos=True, max_token_length=16):
    pure_letters = langCode2alphabet_pure.get(lang, langCode2alphabet_pure['it'])
    minion_group = []
    minion_group_tokens = []
    origin_token_idxs = token2idxs(token, vocab2index, max_token_length, mode='padding')
    new_token = ''
    if mode == "auto":
        for _ in range(size):
            if random.random() <= minion_group_neg_ratio:
                while not len(new_token):
                    new_token, position, action = misspelling_generator_wrapper(token, pure_letters, times, 'none')
                new_token_idxs = token2idxs(new_token, vocab2index, max_token_length, mode='padding')
                if adding_pos:
                    minion_group.append([new_token_idxs, 1, origin_token_idxs, position+1, action])
                else:
                    minion_group.append([new_token_idxs, 1, origin_token_idxs, action])
                minion_group_tokens.append(new_token)
                minion_group_tokens.append([new_token, 1])
            else:
                if adding_pos:
                    minion_group.append([origin_token_idxs, 0, origin_token_idxs, 0, 'origin'])
                else:
                    minion_group.append([origin_token_idxs, 0, origin_token_idxs, 'origin'])
                minion_group_tokens.append([token, 0])

    elif mode == "byrule":
        for _ in range(size):
            if random.random() <= minion_group_neg_ratio:
                while not len(new_token):
                    # avoid empty rule set
                    if lang in rules.keys() and len(rules[lang]):
                        new_token, position, action = misspelling_generator_wrapper(token, pure_letters, times, lang)
                    else:
                        new_token, position, action = misspelling_generator_wrapper(token, pure_letters, times, 'none')
                new_token_idxs = token2idxs(new_token, vocab2index, max_token_length, mode='padding')
                minion_group.append([new_token_idxs, 1, origin_token_idxs, position+1, action])
                minion_group_tokens.append([new_token, 1])
            else:
                minion_group.append([origin_token_idxs, 0, origin_token_idxs, 0, 'origin'])
                minion_group_tokens.append([token, 0])

    return minion_group, minion_group_tokens


def evaluateToken(user, token, method, model_name, batch=1):
    token_lower = token.lower()

    result_kb = 'None'
    global KB_correctionm, vocab2index, activation, model, Type, \
           smoothing, k, length_avg, LM_model
    if KB_correction[token_lower] != '':
        result_kb = decisions[token_lower != KB_correction[token_lower]]
        logging.info("user[{}] [{}][KB] {} correct in {}".format(user, method, token, token_lower == KB_correction[token_lower]))

    if model_name == 'RNN':
        try:
            evaluate_sample = [vocab2index['<SOS>']] + [vocab2index[x] for x in token_lower] + [vocab2index['<EOS>']]
        except KeyError as err:
            logging.info("user[{}][{}][Model={}] Cannot find key of {}".format(user, current_lang, model_name, err))
            evaluate_sample = [vocab2index['<SOS>']] + [vocab2index.get(x, vocab2index['<UNK>']) for x in token_lower] + [vocab2index['<EOS>']]

        sample_scores, sample_predicts, hiddens = evaluate(evaluate_sample, vocab2index, 'rnn', model,
                                                        activation, Type=Type, return_hidden=True)
        sample_scores_show = sample_scores.detach().numpy()
        sample_prob = softmax(sample_scores_show, axis=1)
        result_model = decisions[sample_predicts.tolist()[0]]
        logging.info("user[{}] [{}][Model={}] {} correct in {}".format(user, method, model_name, token, sample_prob.tolist()[0]))

    if model_name == 'CNN':
        try:
            evaluate_sample = [vocab2index['<SOS>']] + [vocab2index[x] for x in token_lower] + [vocab2index['<EOS>']]
        except KeyError as err:
            logging.info("user[{}][{}][Model={}] Cannot find key of {}".format(user, current_lang, model_name, err))
            evaluate_sample = [vocab2index['<SOS>']] + [vocab2index.get(x, vocab2index['<UNK>']) for x in token_lower] + [vocab2index['<EOS>']]

        sample_scores, sample_predicts = evaluate(evaluate_sample, vocab2index, 'cnn', model,
                                                  activation, Type=Type, return_hidden=False)
        sample_scores_show = sample_scores.detach().numpy()
        sample_prob = softmax(sample_scores_show, axis=1)
        result_model = decisions[sample_predicts.tolist()[0]]
        logging.info("user[{}] [{}][Model={}] {} correct in {}".format(user, method, model_name, token, sample_prob.tolist()[0]))

    elif model_name == 'triLM':
        trigramProb = LM_model['trigramProb']
        bigramProb = LM_model['bigramProb']
        threshold_chosen = LM_model['train_threshold']
        # label is not usefull right here
        acc, precision, recall, F1, loglikelihood = languageModelEval([[token_lower, '0']], trigramProb, bigramProb, threshold_chosen, vocab_size=len(index2vocab), 
                                                                        ngram=3, smoothing=smoothing, k=k, lengthAverage=length_avg, show_result=False, return_likelihood=True)
        result_model = decisions[int(loglikelihood[0] <= threshold_chosen)]
        logging.info("user[{}] [{}] [Model={}] {} correct in {} threshold[{}]".format(user, method, model_name, token, np.exp(loglikelihood[0]), np.exp(threshold_chosen)))
    
    else:
        pass

    if result_kb != 'None':
        if result_kb == result_model:
            return result_kb, 'both'
        else:
            return result_kb, 'KB'
    else:
        return result_model, 'model'

@app.teardown_request
def teardown_request(error):
    # record error
    if error is not None:
        print('teardown_request：%s' % error)

@app.route('/')
def index():
    return app.send_static_file('HelloWorld.html'), 400

@app.route('/loadLangModel')
def loadLangModel():
    start_time = time.time()
    lang = request.args.get('lang')
    langCode = lang2code.get(lang, 'NewLing')
    model_name = request.args.get('model')
    if langCode == 'NewLing':
        # Fresh language
        print("Fresh loadcode of {}".format(lang))
        logger.info("Fresh loadcode of {}".format(lang))

        langCode = lang
        lang2code[lang] = lang
        langCode2alphabet[lang] = langCode2alphabet["en"]
        langCode2alphabet_pure[lang] = langCode2alphabet_pure["en"]

        # Vocab based
        rules[lang] = rules["en"]

        # Non-vocab based
        rules[lang] = {}

    print('\nload {} Model ...'.format(lang))
    logger.info('\nload {} Model ...'.format(lang))
    # load model of the language specified
    global vocab2index, index2vocab, model, current_lang, \
           model, optimizer, criterion, LM_model
    current_lang = langCode
    vocab2index, index2vocab = build_vocab(langCode)

    print("vocabulary of size {} ready in {}s".format(len(vocab2index), round(time.time() - start_time, 2)))
    logging.info("update scheduler ready in {}s".format(round(time.time() - start_time, 2)))

    if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true' and TimelyCheck:
        scheduler = BackgroundScheduler()
        scheduler.add_job(func=KB_V_update, trigger="interval", minutes=5)
        scheduler.start()

    print("update scheduler ready in {}s".format(round(time.time() - start_time, 2)))
    logging.info("update scheduler ready in {}s".format(round(time.time() - start_time, 2)))

    global args

    if model_name == 'triLM':
        LM_model = get_model('triLM', len(index2vocab), args)

        # model_dir = './model/' + pattern + '/' + langCode
        model_dir = './model/{}'.format(langCode)

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        model_path = model_dir + '/triLM'

    else:
        model = get_model(model_name.lower(), len(index2vocab), args)
        optimizer = get_optimizer(model, 'adam')
        criterion = get_loss_criterion(loss_criterion)

        optimizers = {'rnn': optimizer}
        models = {'encoder': model}

        # model-CNN/RNN-Pattern-Language
        model_path = './model/{}/'.format(model_name) + pattern + '/' + langCode

    print('try to use {} model'.format(model_name))
    print("model path: {}".format(model_path))
    kb_path = './KnowledgeBase/KnowledgeBase/' + langCode
    vocab_path = './KnowledgeBase/Vocabulary/' + langCode
    if not os.path.isdir(kb_path):
        # New language
        os.mkdir(kb_path)

    else:
        try:
            global KB_correction, KB_size, word_similarity
            with open('./KnowledgeBase/KnowledgeBase/{}/kb.pkl'.format(langCode), 'rb') as fp:
                KB_correction = pickle.load(fp)
                KB_size = len(KB_correction)
            fp.close()
            build_word_similarity(word_similarity, KB_correction)
            logging.info("loading knowledge base of size {} in {}".format(KB_size, lang))
        except:
            KB_size = 0
            print("fresh knowledge base")

    if not os.path.isdir(vocab_path):
        # New language
        os.mkdir(vocab_path)

    else:
        global token2index, index2token, token_size
        try:
            with open('./KnowledgeBase/Vocabulary/{}/token2index.pkl'.format(langCode), 'rb') as fp:
                token2index = pickle.load(fp)
            fp.close()

            with open('./KnowledgeBase/Vocabulary/{}/index2token.pkl'.format(langCode), 'rb') as fp:
                index2token = pickle.load(fp)
            fp.close()

            token_size = len(token2index)
            logging.info("loading vocabulary of size {} in {}".format(token_size, lang))
        except:
            token_size = 0


    if not os.path.isdir(model_path):
        # New language
        os.makedirs(model_path, exist_ok=True)
    else:
        if model_name == 'triLM':
            if os.path.isdir(model_path):
                files = os.listdir(model_path)
                if len(files):
                    # TODO: get the max trainable pkl
                    with open(model_path + '/' + files[0], 'rb') as fp:
                        model = pickle.load(fp)
                    print("load language model at {} threshold: {}".format(files[0], model['train_threshold']))
                else:
                    print("Fresh language model at {}".format(model_path))

        else:
            try:
                load_model_path = model_path + '/best_model.pth.tar' if model_name == 'RNN' else model_path + '/{}/best_model.pth.tar'.format(model_name.lower())
                load_model(model, optimizer, load_model_path, pick_latest=True)
                print("loaded model from {}".format(model_path))
                logger.info("loaded model from " + model_path)
            except Exception as e:
                # new model
                print("err: {}".format(e))
                logger.info("err: {}".format(e))
                print("Fresh model of {} at {}".format(lang, model_path))
                logger.info("Fresh model of {} at {}".format(lang, model_path))
                # pdb.set_trace()

            if USE_CUDA:
                for model in models.values():
                    model.cuda()

    print('{} Model loaded in {}s'.format(lang, round(time.time() - start_time, 2)))
    logger.info('{} Model loaded in {}s'.format(lang, round(time.time() - start_time, 2)))
    result2front = {'lang': lang, 'status': 'succeed', 'VocaSize': token_size, 'KBSize': KB_size,
                    'loadTimose': round(time.time() - start_time, 2),
                    'msg': '{} Model loaded in {}s'.format(lang, round(time.time() - start_time, 2))}
    return jsonify(result2front)

@app.route('/sendNewToken', methods = ['GET', 'POST'])
def sendNewToken():
    if request.method == 'POST':
        lang = request.form['lang']
        token = request.form['Token'].strip("\n").replace("'", '')
        method = request.form['Method']
        user = request.form['user']
        model_name = request.form['model']
        logging.info("user[{}] newToken {} in {}".format(user, token, lang))
        result, channel = evaluateToken(user, token, model_name=model_name, method=method)
        global token2index
        if token2index.get(token, '') == '':
            isOOV = True
        else:
            isOOV = False

        result2front = {'lang': lang, 'Token': token, 'result': result, 'mode':'single-token', 'user':user,
                        'OOV': isOOV, 'channel': channel, 'KB_result': KB_correction.get(token, '[UNK]')}

    return jsonify(result2front)

@app.route('/sendTokenCorrection', methods = ['GET', 'POST'])
def sendTokenCorrection():
    if request.method == 'POST':
        lang = request.form['lang']
        decision = request.form['userDecision']
        user = request.form['user']
        model_name = request.form['model']

        # read boolean value in javascript
        use_batch = request.form['batch'] in ['true', '1']
        if use_batch:
            token = json.loads(request.form['Token'])
            target_token = json.loads(request.form['TargetToken'])
        else:
            token = request.form['Token']
            target_token = request.form['TargetToken']

        candidates = []

        global vocab2index, summary, token2index, minion_group_neg_ratio, stop_mode, \
               top_N_retrieval, KB_correction, max_ckpt_num,  \
               trigramProb, bigramProb, lm_threshold, vocab, LM_model, \
               model, optimizer, criterion

        if decision == "manualCorrection":
            print("user[{}] corrected [{}] to be [{}] in lang[{}]".format(user, token, target_token, lang))
            logger.info("user[{}] corrected [{}] to be [{}] in lang[{}]".format(user, token, target_token, lang))
            KB_correction[token] = target_token
            KB_correction[target_token] = target_token

        # TODO: misspelled is fine to inclcude in the training?
        if decision in ["correct", "correct-misspell", "misspelled", "misspelled-correct", "manualCorrection"]:
            start_time = time.time()
            if type(token) == list:
                isOOV = []
                batch_hard_pad = []
                token_lower = [x.lower().strip(string.punctuation) for x in token]
                target_token_lower = [x.lower().strip(string.punctuation) for x in target_token]

                for x, y in zip(token_lower, target_token_lower):
                    if y != '[None]':
                        if decision not in ['misspelled']:
                            KB_correction[x] = y
                            isOOV_single, volume = addToken(x)
                            isOOV.append(isOOV_single)

                            isOOV_single, volume = addToken(y)
                            isOOV.append(isOOV_single)

                        try:
                            # Tune the model on target words
                            minion_group, minion_group_tokens = build_minion_group(lang, vocab2index, y, minion_group_size,
                                                                                minion_group_neg_ratio, mode='byrule',
                                                                                adding_pos=True)
                            batch_hard_pad.append(minion_group)
                        except Exception as e:
                            print("err in build minion group of SendCorrection at {}： ".format(x) + str(e))
                            pass

                if model_name == 'RNN':
                    batch_hard_pad, summary = tuning_model(model, optimizer, vocab2index,
                                                        (token_lower, token_lower), 0, batch_hard_pad, args=args,
                                                        summary=summary, tune_epoch=tune_epoch, Type=Type, criterion=criterion,
                                                        batch=use_batch, batch_size=batch_size, protect_epoch=0, 
                                                        weights=weights, eval_every_iter=eval_every_iter, 
                                                        hard_sample_stop=hard_sample_stop, early_stop=early_stop,
                                                        dev_set=(), model_save=True, lr_decay=lr_decay, 
                                                        stop_strategy=stop_mode, log=True)

                    print("user[{}][correction][Model={}] tuning model for {} words in {}s".format(user, model_name, len(token_lower), round(time.time() - start_time, 2)))
                    logging.info("user[{}][correction][Model={}] tuning model for {} words in {}s".format(user, model_name, len(token_lower),
                                                                                        round(time.time() - start_time, 2)))

                    prefix='{}-user[{}]'.format(lang, user)
                    model_dir = './model/{}'.format(lang)
                    active_save_model(model, optimizer, model_dir, prefix=prefix, least_recently_used=max_ckpt_num)

                elif model_name == 'triLM':
                    print("previous threshold {}".format(lm_threshold))
                    trigramProb, bigramProb, lm_threshold, vocab_size = languageModelTrain([[x, int(x!=y)] for x, y in zip(token_lower, target_token_lower)], 
                                                                        smoothing=smoothing, k=k, lengthAverage=length_avg, 
                                                                        setting={'setting':'update', 'ngramInfo': {2:bigramProb, 3:trigramProb, 
                                                                        'vocab':list(vocab2index.keys())}, 'loglikelihoods':[], 
                                                                        'tokens':list(KB_correction.keys()), 
                                                                        'tags':[int(key!=value) for key, value in KB_correction.items()]})
                    LM_model = {'trigramProb': trigramProb, 'bigramProb': bigramProb, 'train_threshold': lm_threshold}
                    print("new threshold: {}".format(lm_threshold))
                    vocab += target_token_lower
                    prefix='{}-user[{}]-vocab[{}]'.format(lang, user, len(token2index))
                    model_dir = './model/{}/triLM'.format(lang)
                    language_model_save({'trigramProb': trigramProb, 'bigramProb': bigramProb, 'train_threshold': lm_threshold}, model_dir, prefix=prefix)
                    logging.info("user[{}][correction][Model={}] tuning model for {} words in {}s".format(user, model_name, len(token_lower),
                                                                                                          round(time.time() - start_time, 2)))

                elif model_name == 'CNN':
                    batch_hard_pad, summary = tuning_model(model, optimizer, vocab2index,
                                                            (token_lower, token_lower), 0, batch_hard_pad, args=args,
                                                            summary=summary, tune_epoch=tune_epoch, Type=Type, criterion=criterion,
                                                            batch=use_batch, batch_size=batch_size, protect_epoch=0, 
                                                            weights=weights, eval_every_iter=eval_every_iter, 
                                                            hard_sample_stop=hard_sample_stop, early_stop=early_stop,
                                                            dev_set=(), model_save=True, lr_decay=lr_decay, 
                                                            stop_strategy=stop_mode, log=True)

                    print("user[{}][correction][Model={}] tuning model for {} words in {}s".format(user, model_name, len(token_lower), round(time.time() - start_time, 2)))
                    logging.info("user[{}][correction][Model={}] tuning model for {} words in {}s".format(user, model_name, len(token_lower),
                                                                                        round(time.time() - start_time, 2)))

                    prefix='{}-user[{}]'.format(lang, user)
                    model_dir = './model/{}/{}'.format(model_name, lang)
                    if not os.path.isdir(model_dir):
                        os.makedirs(model_path, exist_ok=True)
                    active_save_model(model, optimizer, model_dir, prefix=prefix, least_recently_used=max_ckpt_num)


            else:
                # receive a single word
                token_lower = token.lower().strip(string.punctuation)
                KB_correction[token_lower] = target_token

                if model_name == 'triLM':
                    trigramProb, bigramProb, lm_threshold, vocab_size = languageModelTrain([[token_lower, 0]], smoothing=smoothing, k=k, lengthAverage=length_avg,
                                                                                            setting={'setting':'update', 'ngramInfo': {2:bigramProb, 3:trigramProb, 'vocab':list(vocab2index.keys())}, 
                                                                                            'loglikelihoods':[], 
                                                                                            'tokens':list(KB_correction.keys()), 
                                                                                            'tags':[int(key!=value) for key, value in KB_correction.items()]})
                    LM_model = {
                        'trigramProb': trigramProb,
                        'bigramProb': bigramProb,
                        'train_threshold':lm_threshold
                    }
                    vocab.append(token_lower)
                    prefix='{}-user[{}]-vocab[{}]'.format(lang, user, len(token2index))
                    model_dir = './model/{}/triLM'.format(lang)
                    language_model_save({'trigramProb': trigramProb, 'bigramProb': bigramProb, 'train_threshold': lm_threshold}, model_dir, prefix=prefix)

                else:
                    batch_hard_pad, batch_group_tokens = build_minion_group(lang, vocab2index, token_lower, minion_group_size,
                                                                            minion_group_neg_ratio, mode='byrule',
                                                                            adding_pos=True)

                    batch_hard_pad, summary = tuning_model(model, optimizer, vocab2index,
                                                        (token_lower, token_lower), 0, minion_group=batch_hard_pad, args=args,
                                                        summary=summary, tune_epoch=tune_epoch, Type=Type, criterion=criterion,
                                                        batch=use_batch, batch_size=batch_size, protect_epoch=0,
                                                        weights=weights, eval_every_iter=eval_every_iter,
                                                        hard_sample_stop=hard_sample_stop, early_stop=early_stop,
                                                        dev_set=(), model_save=True, lr_decay=lr_decay, stop_strategy=stop_mode)
                    print("user[{}][correction][Model={}] tuning model for {} in {}s".format(user, model_name, token_lower, round(time.time() - start_time, 2)))
                    logging.info("user[{}][correction][Model={}] tuning model for {} in {}s".format(user, model_name, token_lower, round(time.time() - start_time, 2)))

                    prefix='{}-user[{}]'.format(lang, user)
                    model_dir = './model/{}'.format(lang)
                    # TODO: add path to the model with model in the dir
                    active_save_model(model, optimizer, model_dir, prefix=prefix, least_recently_used=max_ckpt_num)

                isOOV, volume = addToken(token)


        else:
            volume = removeToken(token)
            candidates = retrieval(token, top_N_retrieval, KB_correction)
            print("{} candidates for {}: {}".format(len(candidates), token, candidates))
            logger.info("{} candidates for {}: {}".format(len(candidates), token, candidates))

            # Use fake samples
            note = request.form['note']
            if note == 'filling':
                global current_lang
                if len(candidates) < top_N_retrieval:
                    pure_letters = langCode2alphabet_pure[current_lang]
                    candidates += [misspelling_generator_wrapper(token, pure_letters, 1, current_lang)[0] for _ in range(10 - len(candidates))]

            candidates = [x for x in candidates if len(x)]

            if token2index.get(token, '') == '':
                isOOV = True
            else:
                isOOV = False

        logging.info("user[{}] thinks token[{}] is decision[{}] in lang[{}]".format(user, token, decision, lang))
        result2front = {'lang': lang, 'Token': token, 'result': candidates, 'mode': 'single-token', 'user':user,
                        'OOV': isOOV, 'VocaSize': volume}
    return jsonify(result2front)

@app.route('/sendRetrievedToken', methods = ['GET', 'POST'])
def sendRetrievedToken():
    if request.method == 'POST':
        lang = request.form['lang']
        method = request.form['Method']
        user = request.form['user']
        use_batch = request.form['use_batch']
        model_name = request.form['model']

        batch_hard_pad = []

        if method == 'retrieveConfirm':
            origin_token = request.form['originalToken']
            origin_token_lower = origin_token.lower()
            retrieved_token = request.form['retrievedToken']
            retrieved_token_lower = retrieved_token.lower()
            logging.info("user[{}][op=retrieve][Model={}] retrieve {} for {} in {}".format(user, model_name, retrieved_token, origin_token, lang))

            global vocab2index, summary, token2index, minion_group_neg_ratio, max_ckpt_num, model, optimizer, criterion, \
                   trigramProb, bigramProb, lm_threshold, vocab, LM_model, KB_correction
            start_time = time.time()

            # Language Model
            if model_name == 'triLM':
                print("previous threshold: {}".format(lm_threshold))
                trigramProb, bigramProb, lm_threshold, vocab_size = languageModelTrain([[origin_token_lower, 1]], smoothing=smoothing, k=k, lengthAverage=length_avg,
                                                                                        setting={'setting':'update', 'ngramInfo': {2:bigramProb, 3:trigramProb, 
                                                                                        'vocab':list(vocab2index.keys())}, 'loglikelihoods':[], 
                                                                                        'tokens':list(KB_correction.keys()), 
                                                                                        'tags':[int(key!=value) for key, value in KB_correction.items()]})
                
                LM_model = {
                    'trigramProb': trigramProb,
                    'bigramProb': bigramProb,
                    'train_threshold':lm_threshold
                }
                print("new_threshold: {}".format(lm_threshold))
                prefix='{}-user[{}]-vocab[{}]'.format(lang, user, len(token2index))
                model_dir = './model/{}/triLM'.format(lang)
                language_model_save({'trigramProb': trigramProb, 'bigramProb': bigramProb, 'train_threshold': lm_threshold}, model_dir, prefix=prefix)

            # Neural Model
            else:
                minion_group, minion_group_tokens = build_minion_group(lang, vocab2index, origin_token_lower, minion_group_size,
                                                                        minion_group_neg_ratio, mode='byrule',
                                                                        adding_pos=True)

                batch_hard_pad.append(minion_group)

                batch_hard_pad, summary = tuning_model(model, optimizer, vocab2index,
                                            (origin_token_lower, retrieved_token_lower), 1, batch_hard_pad, args=args,
                                            summary=summary, tune_epoch=tune_epoch, Type=Type, criterion=criterion,
                                            batch=use_batch, batch_size=batch_size, protect_epoch=0, 
                                            weights=weights, eval_every_iter=eval_every_iter, 
                                            hard_sample_stop=hard_sample_stop, early_stop=early_stop,
                                            dev_set=([], []), model_save=True, lr_decay=lr_decay, stop_strategy=stop_mode)
                logger.info("retrieve model={} for {} in {}s".format(model_name, origin_token, round(time.time() - start_time, 2)))

                prefix='{}-user[{}]'.format(lang, user)
                model_dir = './model/{}'.format(lang)
                active_save_model(model, optimizer, model_dir, prefix=prefix, least_recently_used=max_ckpt_num)

        result2front = {'lang': lang, 'Token': origin_token, 'result': 'retrieve succeed',
                        'mode': 'single-token', 'user':user}
    return jsonify(result2front)


@app.route('/sendNewPattern', methods = ['GET', 'POST'])
def sendNewPattern():
    if request.method == "POST":
        correct_character = request.form['correctCharacter']
        confuse_character = request.form['confuseCharacter']
        lang = request.form['lang']
        langcode = lang2code[lang]
        pattern = '{}/{}'.format(correct_character, confuse_character)

        rules[langcode].append(pattern)

        result2front = {'result': 'succeed', 'currectPattern':rules[langcode], 'pattern': pattern}
    return jsonify(result2front)

@app.route('/receiver', methods = ['GET', 'POST'])
def worker():
    if request.method == "POST":
        # read json + reply
        # data = request.get_json(force=True)
        print("header: ", request.header)
        print("form: ", request.form)
        data = request.get_data()
        result = ''

        # loop over every row
        result += data['Token'] + '(' + data['lang'] + ')' + '\n'
        return result

@app.route('/uploadDocument', methods = ['GET', 'POST'])
def uploadDocument():
    replacement = {'remove_slient_cons':  'removeSlientCons',
                   'double_vowel': 'doubleVowel',
                   'replace_vowel': 'replaceVowel',
                   'delete_vowel': 'deleteVowel'}

    channel_counter = {'both':0, 'KB': 0, 'model':0}

    if request.method in ["POST"]:
        start_time = time.time()
        # read json + reply
        # data = request.get_json(force=True)
        file = request.files['files']
        filename = file.filename
        method = request.form['Method']
        file_type = request.form['type']
        lang = request.form['lang']
        user = request.form['user']
        model_name =  request.form['model']
        max_text_length = int(request.form['maxLength'])
        start = int(request.form['start'])

        result = []
        tokens_submit = []
        isOOV = []

        vocab_increment = 0

        # TODO: whether to use a KB_correction for the model in training
        if filename.endswith(".txt"):
            global token2index, KB_correction
            content = file.read().decode()
            tokens = content.split()
            if file_type == "labelFile":
                n = 5 # 4
                tokens = [tokens[i * n:(i + 1) * n] for i in range((len(tokens) + n - 1) // n)]
                for i, token in enumerate(tokens): # tokens[:start]
                    token_origin = token_preprocessing(token[0])
                    if len(token_origin):
                        try:
                            # KB_correction[token_origin] = token[2]
                            # need_add_vocab, vocab_size = addToken(token_origin)
                            # vocab_increment += need_add_vocab
                            classified_result, channel = evaluateToken(user, token_origin, model_name=model_name, method=method)
                            channel_counter[channel] += 1
                            if classified_result == "correct" and token[1] == '1':
                                result.append('misspell-correct')
                            elif classified_result == "misspelled" and token[1] == '0':
                                result.append('correct-misspell')
                            else:
                                result.append(classified_result)
                            tokens_submit.append(token_origin)
                            isOOV.append(token2index.get(token_origin, '') == '')
                        except:
                            print(i, token_origin)
                        if i > max_text_length:
                            break

            elif file_type == 'textFile':
                for i, token in enumerate(tokens): # tokens[:start]
                    token_origin = token.strip(string.punctuation)
                    if len(token_origin):
                        try:
                            classified_result, channel = evaluateToken(user, token_origin, model_name=model_name, method=method)
                            channel_counter[channel] += 1
                            result.append(classified_result)
                            tokens_submit.append(token)
                            isOOV.append(token2index.get(token_origin, '') == '')
                        except:
                            print(i, token)
                        if i > max_text_length:
                            break

            print("document length: ", len(tokens), " start ", start)

        result2front = {'user': user, 'lang': lang, 'Token': tokens_submit, 'result': result,
                        'mode': 'multi-tokens', 'textLength': i, 'isOOV': isOOV, 'vocabIncrement': vocab_increment,
                        'channel':channel_counter}
        time_consumed = time.time() - start_time
        print("process {} tokens in {}s, avg {} tokens/s".format(i, round(time_consumed, 2), round(i/time_consumed, 2)))
        print(tokens_submit[:10])
        print(result[:10])
        print(Counter(result))
        print("channel: ", channel_counter)

    return jsonify(result2front)

@app.route('/loginValidation', methods = ['GET', 'POST'])
def loginValidation():
    if request.method == "POST":

        print("form: ", request.form)

        user = request.form['user']
        email = request.form['email']
        password = request.form['password']
        method = request.form['Method']

        result = {}
        result['user'] = user

        usrname_file = "./UserInfoBase/userInfo.pkl"
        email_file = "./UserInfoBase/emailInfo.pkl"
        if os.path.isfile(usrname_file):
            with open(usrname_file, "rb") as fp:
                user_data = pickle.load(fp)
            fp.close()

            if user_data.get(user, '') != '':
                if user_data[user].get('password', '') != password:
                    result["status"] = "dupUserName"
                else:
                    result["status"] = "succeed"

            else:
                print("method: ", method, " user: ", user)
                if method == 'register':

                    user_data[user] = {}
                    user_data[user]['email'] = email
                    user_data[user]['password'] = password


                    if os.path.isfile(email_file):
                        with open(email_file, "rb") as fp:
                            email_data = pickle.load(fp)
                        fp.close()

                    else:
                        email_data = {}


                    email_data[email] = {}
                    email_data[email]['user'] = user
                    email_data[email]['password'] = password

                    with open(usrname_file, 'wb+') as fp:
                        pickle.dump(user_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    fp.close()

                    with open(email_file, 'wb+') as fp:
                        pickle.dump(email_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    fp.close()

                    result["status"] = "registrationSucceed"
                else:
                    result["status"] = "needRegistration"

        else:
            user_data = {}
            user_data[user] = {}
            user_data[user]['email'] = email
            user_data[user]['password'] = password

            email_data = {}
            email_data[email] = {}
            email_data[email]['user'] = user
            email_data[email]['password'] = password

            with open(usrname_file, 'wb+') as fp:
                pickle.dump(user_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

            with open(email_file, 'wb+') as fp:
                pickle.dump(email_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

            if method == 'register':
                result["status"] = "registrationSucceed"
            else:
                result["status"] = "newUser"

        return jsonify(result)

@app.route('/collectMetric', methods = ['GET', 'POST'])
def collectMetric():
    if request.method == "POST":
        global misspell2correct, correct2misspell, retrievalEffectiveClick, retrievalManualCorrect, retrievalClick
        misspell2correct = int(request.form['misspell2correct'])
        correct2misspell = int(request.form['correct2misspell'])
        retrievalEffectiveClick = int(request.form['retrievalEffectiveClick'])
        retrievalManualCorrect = int(request.form['retrievalManualClick'])
        retrievalClick = int(request.form['retrievalClick'])
        user = request.form['user']

        result = {}
        result['user'] = user
        try:
            logger.info("user[{}]  misspell2correct: {} correct2misspell: {} retrievalEffectiveClick: {} retrievalManualCorrect: {} retrievalClick: {}".format(
                user, misspell2correct, correct2misspell, retrievalEffectiveClick, retrievalManualCorrect, retrievalClick))
            print("metrics transmission succeed")
            result['status'] = 'succeed'
        except Exception as e:
            print("func[collectMetric] err: {}".format(e))
            result['status'] = 'failed'

        return jsonify(result)

@app.route('/updateMetrics', methods = ['GET', 'POST'])
def updateMetrics():
    if request.method == "POST":
        global accuracy, precision, recall, F1, current_lang, model_name
        accuracy = float(request.form['accuracy'])
        precision = float(request.form['precision'])
        recall = float(request.form['recall'])
        F1 = float(request.form['F1'])
        user = request.form['user']
        model_name = request.form['model']

        if model_name == 'triLM':
            global model
            note = ' threshold[{}]'.format(model['train_threshold'])
        else:
            note = ''

        logger.info("lang[{}] user[{}] model[{}] KB[{}] vocab[{}] accuracy[{}] precision[{}] recall[{}] F1[{}]{}".format(
            current_lang, user, model_name, len(KB_correction), len(token2index), round(accuracy, 2), round(precision, 2), round(recall,2), round(F1, 2), note
        ))   

        result2front = {'user': user, 'model': model_name, 'status': 'succeed'}

        return jsonify(result2front)


if __name__ == '__main__':
    # threaded=True?
    # app.run(host="54.175.62.161", port=8900, debug=True)
    app.run(host="127.0.0.1", port=8900, debug=True)
