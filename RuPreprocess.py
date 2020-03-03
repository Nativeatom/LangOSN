import os
import codecs
import shutil
import chardet
import sys
import pdb
import operator
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import random
import pickle
import string
import itertools
import json
from collections import defaultdict, OrderedDict
import time
import nltk
import pandas as pd
from Model import LuongAttnDecoderRNN, Masked_cross_entropy, CNN
from Preprocessing import preprocess, get_batches, summary_saver, dict_saver
from Data_loader import DataLoader
from Useful_Function import softmax, get_grad, misspelling_generator_wrapper, get_timestamp, extract_timestamp, extract_train_volume
from config import *
from evaluation import metrics_cal
from universial_config import args

import copy

# Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn as nn


class CNNclass(torch.nn.Module):
    def __init__(self, nwords, emb_size, num_filters, window_sizes, dropout, ntags, weight_norm, Type, pretrained_embedding=None):
        super(CNNclass, self).__init__()

        """ layers """
        self.embedding = torch.nn.Embedding(nwords, emb_size)
        if pretrained_embedding is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding).type(Type))
        else:
            # uniform initialization
            torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        # Conv 1d
        self.conv_1d = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_sizes[0],
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv_2d = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_sizes[1],
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv_3d = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_sizes[2],
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.relu = torch.nn.ReLU()
        # Drop out layer
        self.drop_layer = torch.nn.Dropout(p=dropout)
        self.projection_layer = torch.nn.Linear(in_features=3*num_filters, out_features=ntags, bias=True)
        # self.projection_layer = torch.nn.Linear(in_features=num_filters, out_features=ntags, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)
        self.weight_norm = weight_norm
        self.embedding.weight.requires_grad = True

    def forward(self, words, return_activations=False):
        emb = self.embedding(words)                 # nwords x emb_size
        if len(emb.size()) == 3:
            batch = emb.size()[0]
            emb = emb.permute(0, 2, 1)
        else:
            batch = 1
            emb = emb.unsqueeze(0).permute(0, 2, 1)     # 1 x emb_size x nwords

        # emb of size [batch, embedding_size, sentence_length]
        # h of size [batch, filter_size, sentence_length - window_size + 1]
        h1 = self.conv_1d(emb).max(dim=2)[0]
        h2 = self.conv_2d(emb).max(dim=2)[0]
        h3 = self.conv_3d(emb).max(dim=2)[0]
        # h_flat = h1
        h_flat = torch.cat([h1, h2, h3], dim=1)                    # [batch, 3*filter]

        # activation operation receives size of [batch, filter_size, sentence_length - window_size + 1]
        #  activation [batch, sentence_length - window_size + 1] argmax along length of the sentence
        # the max operation reduce the filter_size dimension and select the index ones
        activations = h_flat.max(dim=1)[1]

        # Do max pooling
        h_flat = self.relu(h_flat)
        features = h_flat.squeeze(0)               # [batch, 3*filter]
        h = self.drop_layer(features)
        out = self.projection_layer(h)              # size(out) = 1 x ntags
        if return_activations:
            return out, activations.data.cpu().numpy(), features.data.cpu().numpy()
        return out

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1, ntags=2, bidirection=True,
                 activation='lstm', max_seq_length=16):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirection

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=self.dropout, bias=True, bidirectional=bidirection)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=bidirection)

        self.dense = torch.nn.Linear(in_features=hidden_size, out_features=ntags, bias=True)
        self.position_indicator = torch.nn.Linear(in_features=hidden_size, out_features=max_seq_length, bias=True)

    def forward(self, input_seqs, input_lengths, activation='lstm', padded=True, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        if padded:
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        else:
            packed = embedded
        if activation == 'gru':
            outputs, hidden = self.gru(packed, hidden)

        elif activation == 'lstm':
            outputs, hidden = self.lstm(packed, hidden)
            forward_hidden = hidden[0]
            backward_hidden = hidden[1]
            hidden = forward_hidden + backward_hidden

        if padded:
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)

        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden

    def projection(self, x):
        return self.dense(x)

    def position_projection(self, x):
        return self.position_indicator(x)

    def set_embedding(self, pretrained_embedding):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    def get_embedding(self):
        return self.embedding.weight.data

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1, dropout=0.1, ntags=2,
                 bidirection=True, activation='lstm', max_seq_length=16):
        super(RNN, self).__init__()

        self.name = 'RNN'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = 2 if bidirection else 1

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers=n_layers, bias=True, batch_first=True, bidirectional=bidirection)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=n_layers, bias=True, batch_first=True,
                            dropout=self.dropout, bidirectional=bidirection)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=bidirection)

        self.dense = torch.nn.Linear(in_features=self.bidirectional*hidden_size, out_features=ntags, bias=True)
        self.position_indicator = torch.nn.Linear(in_features=hidden_size, out_features=max_seq_length, bias=True)

    def forward(self, input_seqs, input_lengths, activation='lstm', padded=True, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)


        # One time step
        if activation=='lstm':
            # LSTM
            hidden_state_0 = Variable(
                torch.zeros(self.bidirectional * self.n_layers, input_seqs.size()[0], self.hidden_size))
            cell_state_0 = Variable(
                torch.zeros(self.bidirectional * self.n_layers, input_seqs.size()[0], self.hidden_size))
            outputs, (hidden_state_n, cell_state_n) = self.lstm(embedded, (hidden_state_0, cell_state_0))
        else:
            # vanilla RNN
            # Initialize hidden state with zeros
            hidden_state_0 = Variable(
                torch.zeros(self.bidirectional * self.n_layers, input_seqs.size()[0], self.hidden_size))
            outputs, hidden_state_n = self.rnn(embedded, hidden_state_0)
        return outputs, hidden_state_n

    def projection(self, x):
        return self.dense(x)

    def position_projection(self, x):
        return self.position_indicator(x)

    def set_embedding(self, pretrained_embedding):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    def get_embedding(self):
        return self.embedding.weight.data

    def get_fc(self):
        return self.dense.weight.data

def get_distance(word, target_word):
    # return -1 means discard, non-negative value means agreed
    if word == target_word:
        return 0

    # edit distance is at least 8
    if abs(len(word) - len(target_word)) > 4:
        return -1

    # initial letter doesn't show up in the first 2 letter of the word
    try:
        if target_word[0] not in word[:3] and word[0] not in target_word[:3]:
            try:
                if target_word[1] not in word[:3] and word[1] not in target_word[:3]:
                    return -1
            except:
                pass
    except:
        pass

    return nltk.edit_distance(word, target_word)

def evaluate(data, word2index, model_name, model, activation, Type=torch.LongTensor, return_hidden=False):
    # if len(data) < 16:
    #     data += [word2index['<PAD>']] * (16 - len(data))
    dev_tensor = torch.tensor(data).type(Type)

    if model_name == 'cnn':
        scores = model(dev_tensor.unsqueeze(0))  # batch * 1 * ntags
        scores = scores.unsqueeze(0)
        predicts = scores.argmax(dim=1).cpu().numpy()
        encoder_outputs = None
    if model_name == 'rnn':
        encoder_outputs, encoder_hidden = model(dev_tensor.unsqueeze(0), 1, padded=False, hidden=None)
        encoder_last_outputs = encoder_outputs[:, -1, :]
        scores = model.projection(encoder_last_outputs)
        predicts = scores.argmax(dim=1).cpu().numpy()

    if return_hidden:
        return scores, predicts, encoder_outputs
    else:
        return scores, predicts

def language_model_save(model, path, prefix='', suffix='', least_recently_used=1):
    # replace the model with lowest train volume if the highest is reached
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except Exception as e:
            parent_path = '/'.join(path.split('/')[:-1])
            if not os.path.isdir(parent_path):
                os.mkdir(parent_path)
            os.mkdir(path)

    files = list(os.listdir(path))

    if len(files) >= least_recently_used:
        if len(files) == 1:
            current_train_file = files[0]

        else:
            current_train_vol = int(extract_train_volume(files[0]))
            current_train_file  = files[0]

            for model_file in files[1:]:
                model_train_vol = int(extract_train_volume(model_file))
                if model_train_vol < current_train_vol:
                    current_train_vol = model_train_vol
                    current_train_file = model_file
            
        os.remove(os.path.join(path, current_train_file))

    filename = prefix + '-date[{}]-'.format(get_timestamp().replace(":", "#")) + suffix
    with open(path + '/' + filename + 'triLM.pkl', 'wb') as fp:
        pickle.dump(model, fp, protocol=pickle.HIGHEST_PROTOCOL)
    fp.close()

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("model saved at ", path)

def active_save_model(model, optimizer, path, prefix='', suffix='', least_recently_used=1):
    if not os.path.isdir(path):
        os.mkdir(path)

    files = list(os.listdir(path))

    # replace the oldest model file
    if len(files) >= least_recently_used:
        lru_file = files[0]
        lru_time = extract_timestamp(lru_file)

        for model_file in files[1:]:
            if os.path.isfile(model_file) and model_file[0] != '.':
                model_file_timestamp = extract_timestamp(model_file)
                if model_file_timestamp < lru_time:
                    lru_file = model_file
                    lru_time = extract_timestamp(lru_file)
        try:
            os.remove(os.path.join(path, lru_file))
        except:
            pdb.set_trace()
                
    filename = prefix + '-date[{}]-'.format(get_timestamp().replace(":", "#")) + suffix
    model_parameter = {}
    model_parameter['model'] = model.state_dict()
    model_parameter['optimizer'] = optimizer.state_dict()
    try:
        # : is not allowed in the file name, replace with #
        torch.save(model_parameter, path + '/' + filename+'model.pth.tar')
    except Exception as e:
        print('err: {}'.format(e))
        pdb.set_trace()

def load_model(model, optimizer, path, load_optimizer=True, return_model=False, pick_latest=True):
    if pick_latest:
        dir_path = '/'.join(path.split('/')[:-1])
        start = True
        if os.path.isdir(dir_path):
            for model_file in os.listdir(dir_path):
                if os.path.isfile(os.path.join(dir_path, model_file)):
                    if start:
                        latest_file = model_file
                        latest_timestamp = extract_timestamp(model_file)
                        current_timestamp = latest_timestamp
                        start = False
                    else:
                        current_timestamp = extract_timestamp(model_file)
                        if current_timestamp > latest_timestamp:
                            latest_timestamp = current_timestamp
                            latest_file = model_file
                        
            try:
                path = os.path.join(dir_path, latest_file)
            except:
                print(os.listdir(dir_path))
                pdb.set_trace()

    try:
        model.load_state_dict(torch.load(path))
        print("load model from {}".format(path))
    except:
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model'])
        if load_optimizer:
            optimizer.load_state_dict(ckpt['optimizer'])
            print("successfully load model and optimizer from {}".format(path))
        else:
            print("successfully load model from {}".format(path))

    if return_model:
        return model, optimizer


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def show_sample(data):
    source_word = ''.join([i2v[x] for x in data[0]])
    target_word = ''.join([i2v[x] for x in data[2]])
    if len(data) == 3:
        print("[{} {} {}]".format(source_word, data[1], target_word))
    elif len(data) == 4:
        print("[{} {} {} {}]".format(source_word, data[1], target_word, data[3]))

def word_to_index(word, word2index, max_length, pad=True, replace_dash=True):
    if replace_dash:
        word = word.replace('–', '')
    
    # pay attention to the OOV characters
    indexs = [word2index['<SOS>']] + [word2index.get(x, word2index['<UNK>']) for x in word] + [word2index['<EOS>']]

    if pad:
        if len(indexs) < max_length:
            indexs += [word2index['<PAD>'] for _ in range(max_length - len(indexs))]
        else:
            indexs = indexs[:max_length - 1]
            indexs += [word2index['<EOS>']]

    else:
        if len(indexs) >= max_length:
            indexs = indexs[:max_length - 1]
            indexs += [word2index['<EOS>']]

    return indexs

def position_cal(w1, w2, capital=False, datatype='list'):
    if datatype=='str' and not capital:
        w1 = w1.lower()
        w2 = w2.lower()

    if w1 == w2:
        return 0
    else:
        for position in range(min(len(w1), len(w2))):
            if w1[position] != w2[position]:
                break

        return position

def tuning_model(model, optimizer, word2index, new_word_set, label, minion_group, args, summary, tune_epoch, Type, criterion,
                 number=300, batch=False, early_stop=False, batch_size=10,
                 protect_epoch=0, weights=[0.5, 0.5], eval_every_iter=50,
                 hard_sample_stop=False, dev_set=(), model_save=True, lr_decay=1, acc_gap=0.01, early_break=True,
                 stop_strategy='early_stop', log=False, return_model=False, show_model_status=False):

    # TODO rotate the member in the group
    max_length = 16
    print("Tuning model...")
    if log:
        logger.info("Tuning model...")
    model.train()
    new_word, new_word_correct = new_word_set
    minion_group_size = sum([len(x) for x in minion_group])
    data_size = len(new_word) + minion_group_size
    best_acc = 0
    best_dev_acc = 0
    early_stop_tolerance = 1
    losses = []
    accs = []
    position_accs = []
    test_accs = []
    test_accs_pos = []
    dev_f1s = []
    iter = 0
    keep_training = True
    do_dev = False
    use_batch = batch

    if stop_strategy == 'oneEpoch':
        tune_epoch = 1

    lambda_mspl, lambda_pos = [x/sum(weights) for x in weights]

    if not use_batch:
        input_numerical = word_to_index(new_word, word2index, max_length, pad=True)

        if new_word_correct is not None and new_word != new_word_correct:
            correct_numerical = word_to_index(new_word_correct, word2index, max_length, pad=True)
        else:
            correct_numerical = input_numerical

        # mix the wrong example with the hard samples
        batch_nove = random.sample(minion_group, min(number, len(minion_group)))
        if label==0:
            batch_nove.append([input_numerical, label, correct_numerical, 0, 'tunePair'])
        else:
            for w_index, (x, y) in enumerate(zip(new_word, new_word_correct)):
                if x!=y:
                    break
            batch_nove.append([input_numerical, label, correct_numerical, w_index, 'tunePair'])

        train_tensor_tune = Variable(torch.tensor([x[0] for x in batch_nove]).type(Type))
        # target_tensor_tune = Variable(torch.tensor([x[2] for x in batch_nove]).type(Type))
        random.shuffle(batch_nove)
        input_length_nove = [len(x[0]) for x in batch_nove]

        for epoch in range(tune_epoch):
            optimizer.zero_grad()
            if show_model_status:
                print("epoch {} model training {}".format(epoch, model.training))
            random.shuffle(batch_nove)
            tags_train = [x[1] for x in batch_nove]
            pos_train = [x[3] for x in batch_nove]
            if model.name == 'RNN':
                # RNN
                outputs, hidden = model(train_tensor_tune, input_length_nove, padded=False)
                last_outputs = outputs[:, -1, :]
                scores = model.projection(last_outputs)
                # position_scores = model.position_projection(last_outputs)
            elif model.name == 'CNN':
                # CNN
                scores = model(train_tensor_tune, input_length_nove, padded=False)

            predicts = scores.argmax(dim=1).cpu().numpy()
            # position_scores = model.position_projection(encoder_last_outputs)
            # position_predicts = position_scores.argmax(dim=1).cpu().numpy()
            # train_correct += sum(predicts == tags_train)
            # train_correct_negative += sum([x * y for x, y in zip(predicts, tags_train)])
            my_loss = lambda_mspl * criterion(scores, torch.tensor(tags_train).type(Type)) 
            # multitasking
                    #     + \
                    #   lambda_pos * criterion(position_scores, torch.tensor(pos_train).type(Type))
            my_loss.backward()
            optimizer.step()
            iter += 1

        tuning_summary = "Finish tuning of {}\navg_loss: {} acc: {} volume: {}".format(new_word,
                                                                            round(my_loss.item() / len(batch_nove), 6),
                                                                            round(sum(predicts == tags_train) / len(batch_nove), 4),
                                                                            len(batch_nove))
        print(tuning_summary)
        if log:
            logger.info(tuning_summary)

    else:
        input_numerical = [word_to_index(x, word2index, max_length, pad=True) for x in new_word]
        batch_nove = list(itertools.chain.from_iterable(minion_group))

        if new_word_correct is not None and new_word != new_word_correct:
            correct_numerical = [word_to_index(x, word2index, max_length, pad=True) for x in new_word_correct]

            # TODO: add calculation of position
            batch_nove += [[input_, label, correct_, position_cal(input_, correct_, False, 'list'), 'tunePair'] for input_, correct_ in zip(input_numerical, correct_numerical)]

        else:
            correct_numerical = input_numerical
            batch_nove += [[x, label, x, 0, 'tunePair'] for x in input_numerical]

        # batch_size = 10
        num_of_batches = int(len(batch_nove) / batch_size)

        hard_samples = []
        inconfident_number = []
        best_acc = 0

        # with open('./files/eng_TOEFL/highrate/en_most_frequent_200_in_whole_oov_100_2.txt', 'r') as fp:
        #     test = fp.readlines()
        #     test = [x.strip('\n').split() for x in test]


        # has dev words and avoid no input in the for the dev
        if len(dev_set) and len(dev_set[0]) > 0:
            do_dev = True
            test_words, test_tags, test_pos_tags = dev_set
            test_tensor = Variable(torch.tensor([word_to_index(x, word2index, max_length, pad=True) for x in test_words]).type(Type))

        epoch_loss = 0
        for epoch in range(tune_epoch):
            if show_model_status:
                print("epoch {} model training {}".format(epoch, model.training))

            # TODO: uncomment this
            # random.shuffle(batch_nove)
            # pdb.set_trace()/
            acc = 0
            position_acc = 0
            # epoch_loss = 0
            if not keep_training:
                break
            for i in range(num_of_batches):
                if not keep_training:
                    break
                batch = batch_nove[i * batch_size: (i + 1) * batch_size]
                input_lengths = [len(x[0]) for x in batch]
                max_input_length = max(input_lengths) + 2
                max_target_length = max([len(x[2]) for x in batch]) + 2
                train_tensor = Variable(torch.tensor([x[0] for x in batch]).type(Type))
                target_tensor = Variable(torch.tensor([x[2] for x in batch]).type(Type))
                optimizer.zero_grad()
                tags_train = [x[1] for x in batch]
                pos_train = [x[3] for x in batch]

                # encoder_outputs of [batch, max_seq_len, hidden_size]
                # encoder_hidden of [2*layer, max_seq_len, hidden_size]
                if model.name == 'RNN':
                    encoder_outputs, encoder_hidden = model(train_tensor, input_lengths, padded=False)
                    encoder_last_outputs = encoder_outputs[:, -1, :]
                    # TODO: auxiliary task of autoencoding?
                    # decoder_input = Variable(torch.LongTensor([v2i['<UNK>']] * batch_size))
                    # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
                    scores = model.projection(encoder_last_outputs)
                    position_scores = model.position_projection(encoder_last_outputs)
                    position_predicts = position_scores.argmax(dim=1).cpu().numpy()

                else:
                    scores = model(train_tensor, input_lengths, padded=False)
                    position_predicts = [0 for _ in batch]
                    

                predicts = scores.argmax(dim=1).cpu().numpy()
                scores_prob = softmax(scores.detach().numpy(), axis=1)


                # TODO: Collect examples hard to train
                inconfident_indexes_high = np.where(scores_prob[:, 1] > 0.45)[0].tolist()
                inconfident_indexes_low = np.where(scores_prob[:, 1] < 0.55)[0].tolist()
                inconfident_indexes = [x for x in inconfident_indexes_low if x in inconfident_indexes_high]
                inconfident_number.append(len(inconfident_indexes))
                # hard_samples[epoch] = hard_samples[epoch] + [batch[x] for x in inconfident_indexes]

                # TN_indexes = set(np.where(predicts == 0)[0]) & set(np.where(np.array(tags_train) == 1)[0])
                # TP_indexes = np.where((predicts == tags_train) == True)[0]

                # for TN_index in TN_indexes:
                #     TN_dict[''.join(
                #         [i2v[x] for x in batch[TN_index][0] if x not in [v2i['<PAD>'], v2i['<SOS>'], v2i['<EOS>']]])] += 1
                #
                # for TP_index in TP_indexes:
                #     TP_dict[''.join(
                #         [i2v[x] for x in batch[TP_index][0] if x not in [v2i['<PAD>'], v2i['<SOS>'], v2i['<EOS>']]])] += 1

                # train_correct += sum(predicts == tags_train)
                # train_correct_negative += sum([x * y for x, y in zip(predicts, tags_train)])
                if lambda_pos != 0:
                    my_loss = lambda_mspl * criterion(scores, torch.tensor(tags_train).type(Type)) + \
                              lambda_pos * criterion(position_scores, torch.tensor(pos_train).type(Type))
                else:
                    my_loss = criterion(scores, torch.tensor(tags_train).type(Type))
                my_loss.backward()
                optimizer.step()

                acc += sum(predicts == tags_train)
                position_acc += sum([x==y for x, y in zip(position_predicts, pos_train)])
                epoch_loss += my_loss.item()

                iter += 1
                if do_dev and iter % eval_every_iter == 0:
                    div = eval_every_iter
                    model.eval()
                    if args.model in ['rnn', 'birnn']:
                        outputs_eval, hidden_eval = model(test_tensor, len(test_words), padded=False)
                        last_outputs_eval = outputs_eval[:, -1, :]
                        # TODO: auxiliary task of autoencoding?
                        # decoder_input = Variable(torch.LongTensor([v2i['<UNK>']] * batch_size))
                        # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
                        scores_eval = model.projection(last_outputs_eval)
                        position_scores_eval = model.position_projection(last_outputs_eval)
                        position_predicts_eval = position_scores_eval.argmax(dim=1).cpu().numpy()
                    else:
                        scores_eval, outputs_data, features = model(test_tensor, len(test_words), return_activations=True, padded=False)
                        position_predicts_eval = [0 for _ in range(test_tensor.shape[0])]

                    predicts_eval = scores_eval.argmax(dim=1).cpu().numpy()
                    test_acc = sum([x == y for x, y in zip(predicts_eval, test_tags)])


                    test_position_acc = sum([x == y for x, y in zip(position_predicts_eval, test_pos_tags)])
                    model.train()

                    test_accuracy, _,  _, f1 = metrics_cal(predicts=predicts_eval, tags=test_tags, detail=False)
                    losses.append(epoch_loss / (div*batch_size))
                    accs.append(acc / (div*batch_size))
                    test_accs.append(test_accuracy)
                    test_accs_pos.append(test_position_acc / len(test_words))
                    dev_f1s.append(f1)
                    epoch_log = "[Epoch {}][Iter {}] avg_loss: {} acc: {} dev acc: {} f1: {} pos: train {} dev {} volume: {}".format(epoch,
                                                                iter,
                                                                round(epoch_loss / (div*batch_size), 4),
                                                                round(acc / data_size, 4),
                                                                round(test_accuracy, 4),
                                                                round(f1, 4),
                                                                round(position_acc / data_size, 4),
                                                                round(test_position_acc / len(test_words), 4),
                                                                len(batch_nove))
                    print(epoch_log)

                    if log:
                        logger.info(epoch_log)

                    if hard_sample_stop and len(inconfident_indexes) == 0:
                        keep_training = False
                        print("[iter {}]Empty hard sample ....".format(iter))
                        break

                    if test_acc / len(test_words) >= best_dev_acc:
                        best_dev_acc = test_acc / len(test_words)

                    elif test_acc / len(test_words) < best_dev_acc - acc_gap and \
                            early_stop and epoch > protect_epoch:
                        keep_training = False
                        print(test_acc / len(test_words), best_dev_acc, test_acc / len(test_words) < best_dev_acc - acc_gap)
                        if log:
                            logger.info(test_acc / len(test_words), best_dev_acc, test_acc / len(test_words) < best_dev_acc - acc_gap)
                        early_stop += 1
                        if early_stop > early_stop_tolerance:
                            print("[iter{}][lr={}]Early stopping ...".format(iter, optimizer.param_groups[0]['lr']))
                            if log:
                                logger.info("[iter{}][lr={}]Early stopping ...".format(iter, optimizer.param_groups[0]['lr']))
                            # keep_training = False
                            early_stop = 1
                            best_dev_acc = test_acc / len(test_words)

                            for param_group in optimizer.param_groups:
                                curr_lr = param_group['lr']
                                param_group['lr'] = curr_lr*lr_decay

                            if model_save:
                                name = './model/incrementalTraining/{}_vol{}_batch{}_epoch{}_iter{}_devAcc{}_devF1{}_lr{}.pth.tar'.format(
                                        summary['langcode'], len(new_word), batch_size, epoch, iter,
                                        round(test_accuracy, 3), round(f1, 3), curr_lr)
                                save_model(model, name)

                            if early_break:
                                break

                    epoch_loss = 0
        
        if type(new_word) == list:
            try:
                print("Finish tuning of {} tokens like {}".format(len(new_word), random.choice(new_word)))
            except:
                pdb.set_trace()
            if log:
                logger.info("Finish tuning of {} tokens like {}".format(len(new_word), random.choice(new_word)))
            
        elif type(new_word) == str:
            print("Finish tuning of {} tokens like {}".format(1, new_word))
            
            if log:
                logger.info("Finish tuning of {} tokens like {}".format(1, new_word))
                
        else:
            pdb.set_trace()

    model.eval()
    summary['loss'].append(losses)
    summary['accuracy'].append(accs)
    if do_dev:
        summary['dev_acc'].append(test_accs)
        summary['dev_f1'].append(dev_f1s)
        summary['dev_acc_pos'].append(test_accs_pos)
    summary['trigger'].append(new_word)
    summary['protect_epoch'] = protect_epoch
    summary['epoch_stop'] = epoch
    if return_model:
        return minion_group, summary, model
    else:
        return minion_group, summary

class ActionCollector():
    def __init__(self, lang):
        self.collector = {'succeed': defaultdict(list), 'fail': defaultdict(list), 'total':defaultdict(int)}
        self.step_collector = {'succeed': defaultdict(float), 'fail': defaultdict(float)}
        self.total_num = defaultdict(int)
        self.lang = lang

    def newCollector(self):
        self.step_collector = {'succeed': defaultdict(float), 'fail': defaultdict(float)}

    def merge(self):
        for key in list(self.step_collector['succeed'].keys()) + list(self.step_collector['fail'].keys()):
            total_num = self.step_collector['succeed'][key] + self.step_collector['fail'][key]
            self.total_num[key] = int(total_num)
            self.collector['succeed'][key].append(self.step_collector['succeed'][key] / total_num)
            self.collector['fail'][key].append(self.step_collector['fail'][key] / total_num)
            self.step_collector['succeed'][key] = self.step_collector['succeed'][key] / total_num
            self.step_collector['fail'][key] = self.step_collector['fail'][key] / total_num

def result_analysis(result):
    FP = 0
    TP = 0
    TN = 0
    FN =  0
    check_list = 0
    dev_hit = 0
    unseen = 0
    total_num = len(result)

    for key, in result.keys():
        decision = result[key]['decision']
        golden = result[key]['golden']
        train_hist = result[key]['dict']
        dev_hist = result[key]['dev_hit']

        if (decision - golden == 1):
            FN += 1
        if golden - decision == 1:
            FP += 1
        if decision == golden:
            TN += decision
            TP += (1-decision)
        if train_hist:
            check_list += 1
        if dev_hist:
            dev_hit += 1
        if train_hist+dev_hist == 0:
            unseen += 1

    print("Total num: ", total_num)
    print("misclassif correct spelling(FN): {}({})".format(FN, FN/total_num))
    print("corrclassif misspelling(TN): {}({})".format(TN, TN / total_num))
    print("corrclassif correct spelling(TP): {}({})".format(TP, TP / total_num))
    print("misclassif misspelling(FP): {}({})".format(FP, FP / total_num))
    print("list check(train): {}({})".format(train_hist, train_hist / total_num))
    print("list check(train): {}({})".format(dev_hist, dev_hist / total_num))
    print("unseen: ", unseen)

if __name__ == "__main__":
    global i2v
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pattern", type=str, help="training patter", default="random_zero_shot",
                        choices=["normal", 'random_zero_shot', 'mixed'])
    parser.add_argument("-m", "--mode", type=str, help="mode of translation", default="real",
                        choices=["gan", 'hash', 'real'])
    parser.add_argument("-err_m", "--err_mode", type=str, help="mode of generate misspelling", default="auto",
                        choices=["auto", 'byrule'])
    parser.add_argument("-train_err_m", "--train_err_mode", type=str, help="mode of generate misspelling for training data",
                        default="auto",
                        choices=["auto", 'byrule'])
    parser.add_argument("-testm", "--test_mode", type=str, help="mode of test", default="real",
                        choices=["fake", 'real', 'all_real', 'unseen'])
    parser.add_argument("-testnum", "--test_num", type=int, help="number of test sample", default=2000)
    parser.add_argument("-data", "--data", type=str, help="language of data", default="ru",
                        choices=["ru", 'en',
                                 'fin', 'fin_wiki', 'fin_wiki_rule',
                                 'ru_btl', 'ru_wiki', 'ru_wiki_rule',
                                 'ita', 'ita_wiki', 'ita_wiki_rule',
                                 'spa', 'spa_wiki', 'spa_wiki_rule',
                                 'trk', 'trk_wiki', 'trk_wiki_rule'])
    parser.add_argument("-ml", "--model", type=str, help="model used for prediction", default="logistic",
                        choices=["logistic", 'cnn', 'seq2seq', 'rnn-logistic', 'rnn'])
    parser.add_argument("-loss", "--loss", type=str, help="loss function", default="CrossEntropy",
                        choices=["CrossEntropy", "mse"])
    parser.add_argument("-act", "--activation", type=str, help="activation module", default="lstm",
                        choices=["lstm", "gru"])
    parser.add_argument("-estop", "--early_stop", type=str, help="mode of early stopping", default="immediate",
                        choices=["no", "immediate", "tolerate"])
    parser.add_argument("-proj", "--projection", type=str, help="projection method", default="ann",
                        choices=["ann", "std"])
    parser.add_argument("-tr", "--ratio", type=float, help="ratio of training data", default=1)
    parser.add_argument("-tnratio", "--test_neg_ratio", type=float, help="ratio of negative samples in test data", default=0.5)
    parser.add_argument("-mask", "--mask", type=bool, help="whether use mask for loss calculation", default=False)
    parser.add_argument("-bidir", "--bidirectional", type=bool, help="whether using bidirectional neural cells", default=False)
    parser.add_argument("-zsnegratio", "--shot_neg_ratio", type=float, help="ratio of negative samples in few shot", default=0.3)
    parser.add_argument("-lyr", "--layers", type=int, help="layers of network", default=1)
    parser.add_argument("-zssample", "--shot_sample", type=int, help="number of samples for zero shot", default=20)
    parser.add_argument("-sdw", "--seed_words", type=int, help="number of seed words in the training", default=20)
    parser.add_argument("-sd", "--seed", type=int, help="seed of randomness", default=0)
    parser.add_argument("-weight", "--weight_norm", type=float, help="weight norm to be intialized", default=1)
    parser.add_argument("-hdd", "--hidden_size", type=int, help="hidden size of cells", default=50)
    parser.add_argument("-embedding", "--embedding_size", type=int, help="embedding size", default=30)
    parser.add_argument("-filt", "--filter", type=int, help="filter_size", default=8)
    parser.add_argument("-d", "--dropout", type=float, help="dropout probability", default=0.1)
    parser.add_argument("-opt", "--opt", type=str, help="optimizer", default='sgd', choices=["sgd", "adam"])
    parser.add_argument("-wdecay", "--weight_decay", type=float, help="weight decay of optimizer", default=0)
    parser.add_argument("-lr", "--lr", type=float, help="learning rate", default=0.001)
    parser.add_argument("-bth", "--batch", type=int, help="batch size of training", default=30)
    parser.add_argument("-ep", "--epoch", type=int, help="epoch of training", default=15)
    parser.add_argument("--load_model", type=int, help="whether load the best model", default=1)
    parser.add_argument("-eval", "--eval", type=int, help="evaluation iteration", default=30)
    parser.add_argument("--save_model", type=bool, help="whether to save the model parameters", default=False)
    parser.add_argument("--save_result", type=int, help="whether to save the model parameters", default=1)
    parser.add_argument("--save_decode", type=bool, help="whether to save the decode sequence", default=False)
    # TODO Special test case
    args = parser.parse_args()

    action_collecter = ActionCollector(args.data)
    start_time = time.time()
    if args.data == 'ru':
        with codecs.open('./SourceSent_Ru.txt', 'r', 'utf-8') as fp:
            text = fp.readlines()
            text = [x.strip('\n').strip('\r') for x in text]

        with codecs.open('./TargetSent_Ru.txt', 'r', 'utf-8') as fp:
            Target_text = fp.readlines()
            Target_text = [x.strip('\n').strip('\r') for x in Target_text]

        i2v = {i+1: glagolitsa[i] for i in range(len(glagolitsa))}
        i2v[0] = '<PAD>'
        i2v[len(i2v)] = ' '
        i2v[len(i2v)] = '-'
        i2v[len(i2v)] = '<UNK>'
        i2v[len(i2v)] = '<EOS>'
        i2v[len(i2v)] = '<SOS>'
        v2i = {v: k for k, v in i2v.items()}
        target = []

        for i in range(len(Target_text)):
            target.append((i, Target_text[i]))

        result = []
        for i in range(len(text)):
            result.append((i, text[i]))
        source_tags = [x[0] for x in result]
        target_tags = [x[0] for x in target]
        common_indexes = [x for x in source_tags if x in target_tags]
        nums = len(common_indexes)
        random.shuffle(common_indexes)
        train_indexes = common_indexes[:int(0.85*nums)]
        dev_indexes = common_indexes[int(0.85*nums)+1:]
        if args.ratio < 1:
            train_indexes = train_indexes[:int(args.ratio*len(train_indexes))]
        target_train = [x[1].lower().replace('-', '') for x in target if x[0] in train_indexes]
        source_train = [preprocess(x[1]) for x in result if x[0] in train_indexes]
        target_dev = [x[1].lower().replace('-', '') for x in target if x[0] in dev_indexes]
        source_dev = [preprocess(x[1]) for x in result if x[0] in dev_indexes]

        train_words = list(' '.join(target_train + source_train).split())
        train_words_set = list(set(train_words))

        correct_words_train = ' '.join(target_train).split()
        correct_words_dev = ' '.join(target_train).split()

        pure_letters = glagolitsa

    if args.data == "en":
        with codecs.open('./files/eng_TOEFL/train.txt', 'r', 'utf-8') as fp:
            train_raw = fp.readlines()
            train_raw = [x.strip('\n').strip('\r') for x in train_raw]

        with codecs.open('./files/eng_TOEFL/dev.txt', 'r', 'utf-8') as fp:
            Target_dev = fp.readlines()
            Target_dev = [x.strip('\n').strip('\r') for x in Target_dev]

        with codecs.open('./files/eng_TOEFL/test.txt', 'r', 'utf-8') as fp:
            Target_test = fp.readlines()
            Target_test = [x.strip('\n').strip('\r') for x in Target_test]

        pure_letters = ' '.join(ascii_lowercase).split()

        i2v = {i+1: letters[i] for i in range(len(letters))}
        i2v[0] = '<PAD>'
        i2v[len(i2v)] = ' '
        i2v[len(i2v)] = '-'
        i2v[len(i2v)] = '<UNK>'
        i2v[len(i2v)] = '<EOS>'
        i2v[len(i2v)] = '<SOS>'
        v2i = {v: k for k, v in i2v.items()}

        train_p = [preprocess(x.lower().replace('-', '').replace('´', '').replace('”','').replace('“','')).split() for x in train_raw]
        train_p = [x for x in train_p if len(x) == 3]

        # [w1, w1!=w2, w2]
        source_train = [x[0] for x in train_p]
        target_train = [x[2] for x in train_p]
        train = []

        for x in train_p:
            try:
                w1 = [v2i[k] for k in x[0]]
                w2 = [v2i[k] for k in x[2]]
                train.append([w1, int(w1 != w2), w2])
            except:
                pass

        # train = [[[v2i[k] for k in x[0]], 0, [v2i[k] for k in x[2]]] for x in train if len(x) == 3]
        dev_batches_p = [preprocess(x.lower().replace('-', '').replace('´', '').
                                    replace('”','').replace('“','')).split() for x in Target_test]
        dev_batches_p = [x for x in dev_batches_p if len(x) >= 3]
        source_dev = [x[0] for x in dev_batches_p]
        target_dev = [x[2] for x in dev_batches_p]
        dev_batches = []

        negative_ratio = args.test_neg_ratio
        dev_batches_p_num = len(dev_batches_p)
        if args.err_mode == 'byrule':
            for x in dev_batches_p:
                try:
                    if random.randint(1, dev_batches_p_num) / dev_batches_p_num < negative_ratio:
                        w1, action = misspelling_generator_wrapper(x[2], pure_letters, 1, 'en')
                        w1_numer = [v2i[k] for k in w1]
                        w2 = [v2i[k] for k in x[2]]
                        dev_batches.append([w1_numer, int(w1 != w2), w2, action])
                    else:
                        w2 = [v2i[k] for k in x[2]]
                        dev_batches.append([w2, 0, w2, 'origin'])
                except:
                    pass
        else:
            for x in dev_batches_p:
                try:
                    w1 = [v2i[k] for k in x[0]]
                    w2 = [v2i[k] for k in x[2]]
                    dev_batches.append([w1, int(w1 != w2), w2, 'auto'])
                except:
                    pass

        print("#dev raw {} negative {}".format(len(dev_batches), len([x for x in dev_batches if x[1]==1])))

        # dev_batches = [[[v2i[k] for k in x[0]], 1, [v2i[k] for k in x[2]]] for x in dev_batches if len(x) == 3]
        if args.ratio < 1:

            # Needs to shuffle, unless cannot sample correct ones
            random.shuffle(train)
            train = train[:int(args.ratio*len(train))]
            train_p = train_p[:int(args.ratio*len(train_p))]

        train_words = [x[0] for x in train_p] + [x[2] for x in train_p]
        train_words_set = list(set([x[0] for x in train_p]).union(set([x[2] for x in train_p]), set([x[0] for x in dev_batches_p])))

        correct_words_train = list(set([x[1] for x in train_p]))
        correct_words_dev = list(set([x[0] for x in dev_batches_p]))

    else:
        train_raw, Target_test, test_file, i2v, v2i, pure_letters, letters = DataLoader(args.data, letters)

        train_p = [preprocess(x.lower().replace('´', '').replace('"', '')).split() for x in train_raw]
        train_p = [x for x in train_p if len(x) >= 3]

        # [w1, w1!=w2, w2]
        source_train = [x[0] for x in train_p]
        target_train = [x[2] for x in train_p]
        train = []

        discard = 0
        for x in train_p:
            try:
                w1 = [v2i[k] for k in x[0]]
                w2 = [v2i[k] for k in x[2]]
                train.append([w1, int(w1 != w2), w2])
            except:
                discard += 1
                pass

        print("Train Discard: {}/{}".format(discard, len(train)))
        # train = [[[v2i[k] for k in x[0]], 0, [v2i[k] for k in x[2]]] for x in train if len(x) == 3]
        dev_batches_p = [preprocess(x.lower().replace('´', '')).split() for x in Target_test]
        dev_batches_p = [x for x in dev_batches_p if len(x) >= 3]
        source_dev = [x[0] for x in dev_batches_p]
        target_dev = [x[2] for x in dev_batches_p]
        dev_batches = []

        discard = 0
        if args.err_mode == 'byrule':
            dev_batches_p = [x for x in dev_batches_p if len(x) == 4]
            actions_dev = [x[3] for x in dev_batches_p]
            for x in dev_batches_p:
                try:
                    w1 = [v2i[k] for k in x[0]]
                    w2 = [v2i[k] for k in x[2]]
                    dev_batches.append([w1, int(w1 != w2), w2, x[3]])
                except:
                    discard += 1
                    pass
        else:
            for x in dev_batches_p:
                try:
                    w1 = [v2i[k] for k in x[0]]
                    w2 = [v2i[k] for k in x[2]]
                    dev_batches.append([w1, int(w1 != w2), w2, 'auto'])
                except:
                    discard += 1
                    pass


        print("Dev Discard: {}/{}".format(discard, len(dev_batches)))
        if args.ratio < 1:
            # Needs to shuffle, unless cannot sample correct ones
            random.shuffle(train)
            train = train[:int(args.ratio * len(train))]
            train_p = train_p[:int(args.ratio * len(train_p))]

        train_words = [x[0] for x in train_p] + [x[2] for x in train_p]
        train_words_set = list(set([x[0] for x in train_p]).union(set([x[2] for x in train_p])))
        total_words_set = list(
            set([x[0] for x in train_p]).union(set([x[2] for x in train_p]), set([x[0] for x in dev_batches_p])))

        correct_words_train = list(set([x[2] for x in train_p]))
        correct_words_dev = list(set([x[0] for x in dev_batches_p]))

    word_freq_train = defaultdict(float)
    word_freq_dev = defaultdict(float)

    train_words_num = len(train_words)
    print("Training {} words time {}s".format(train_words_num, round((time.time() - start_time) / 60, 4)))

    # TODO Shared vocabulary of training and test set
    # shard_words = list(set([x for x in correct_words_dev if x in correct_words_train]))


    # with open('./summary/missplled_history.pkl', 'rb') as fp_hist:
    # 	old_error_dict = pickle.load(fp_hist)
    # fp_hist.close()
    # print('Old errors #: ', len(old_error_dict))
    new_misp_error_dict = defaultdict(int)

    # Parameters
    NEPOCH = args.epoch
    batch_size = args.batch
    if args.data == "ru":
        batch_num = int(len(source_train) / batch_size)
    if args.data == "en":
        batch_num = int(len(train) / batch_size)
    else:
        batch_num = int(len(train) / batch_size)
    hidden_size = args.hidden_size
    dropout = args.dropout
    learning_rate = args.lr
    n_layers = args.layers
    embedding_size = args.embedding_size
    decoder_learning_ratio = 5.0
    USE_CUDA = False
    attn_model = 'dot'
    best_model = 0
    best_model_parameter = {'model':{}, 'optimizer':{}}
    best_model_metrics = {'train_acc':0, 'eval_acc':0, 'best_epoch':0,
                            'acc':0, 'neg_acc':0, 'precision':0,
                            'recall':0, 'F1':0}
    best_f1 = 0
    best_correct = []
    best_wrong = []
    KB_correction = defaultdict(str)
    Type = torch.LongTensor

    nwords = len(i2v)
    EMB_SIZE = len(i2v)
    ntags = 2
    bidirection = False
    # use one-hot embedding for training
    pretrained_embedding = np.identity(len(i2v))

    # samples with confidence less than 60%
    hard_samples = defaultdict(list)

    words_all_fake_misspl = []

    # CNN Parameter
    WIN_SIZE = [5, 6, 7]
    WIN_SIZE_MAX = max(WIN_SIZE)

    if args.model == 'cnn':
        FILTER_SIZE = args.filter
        model = CNN(nwords, EMB_SIZE, FILTER_SIZE, WIN_SIZE, dropout, ntags, 1, Type, pretrained_embedding)
        if args.opt == 'sgd':
            if args.weight_decay > 0:
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        if args.opt == 'adam':
            if args.weight_decay > 0:
                optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.Adam(model.parameters())

        optimizers = {'cnn': optimizer}
        models = {'cnn': model}

    if args.model == 'rnn':
        WIN_SIZE_MAX = 16

        # Initialize models
        decoder_output_size = {'pinyin': len(i2v)}
        encoder = RNN(len(i2v), hidden_size, embedding_size, n_layers=n_layers, dropout=dropout,
                      ntags=2, bidirection=bidirection)

        # Initialize optimizers and criterion
        if args.opt == 'sgd':
            if args.weight_decay > 0:
                optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)

        if args.opt == 'adam':
            if args.weight_decay > 0:
                optimizer = optim.Adam(encoder.parameters(), weight_decay=args.weight_decay)
            else:
                optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

        optimizers = {'rnn': optimizer}
        models = {'encoder': encoder}
        embedding_original = encoder.get_embedding()

    # decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if args.loss == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    if args.loss == "mse":
        criterion = nn.MSELoss()

    # Move models to GPU
    if USE_CUDA:
        for model in models.values():
            model.cuda()

    print('Dev statistics')
    if args.data == "ru":
        dev_batches = list(get_batches(source_dev, target_dev, batch_size, v2i,
                                        padding_mode='max', level='binary', to_yield=False, return_ratio=True))
    if args.data == "en":
        pass

    dev_word_distribution = defaultdict(list)

    # Get edit distance distribution of dev set
    dev_edit_dist_distribution = defaultdict(int)

    for x in dev_batches:
        if int(x[1]) == 1:
            edistance = nltk.edit_distance(''.join(i2v[l] for l in x[0]), ''.join(i2v[l] for l in x[2]))
            dev_word_distribution[''.join(i2v[l] for l in x[2])].append([''.join(i2v[l] for l in x[0]), edistance])
            dev_edit_dist_distribution[edistance] += 1

    start_time = time.time()
    if args.pattern == "random_zero_shot":
        capacity = min(2*args.seed_words, 50)
        if args.data != "en":
            if 2 * args.seed_words < len(train_words_set):
                seed_words = random.sample(list(train_words_set), 2 * args.seed_words)
            else:
                seed_words = [random.choice(list(train_words_set)) for _ in
                              range(min(2 * args.seed_words))]

        else:
            dev_distri_num = len(dev_word_distribution)
            if 2 * args.seed_words < len(train_words_set):
                seed_words = random.sample(list(train_words_set), 2 * args.seed_words)
            else:
                seed_words = [random.choice(list(train_words_set)) for _ in
                              range(min(2 * args.seed_words))]
            # seed_words = [random.choice(list(dev_word_distribution.keys())) for _ in range(min(2*args.seed_words, dev_distri_num))]

        seed_words = [x for x in seed_words if len(x) >= 3]
        seed_words_selected = []
        words_similarity = {}
        calDist_time = time.time()
        seed_stack = seed_words.copy()
        finish = 0

        langCode = data2langcode[args.data]
        # select the seed word set
        while finish < args.seed_words:
            try:
                word = seed_stack[-1]
            except:
                pdb.set_trace()

            # word_similarity edit_distance:list of words
            word_similarity = defaultdict(list)
            count = 0
            for w in train_words_set:
                distance = get_distance(w, word)
                if distance > 0:
                    word_similarity[distance].append(w)
                    count += 1
                    if count > capacity:
                        break

            if len(word_similarity) > 0:
                finish += 1
                seed_words_selected.append(word)
            else:
                # No match in the vocabulary, using fake edition to fill up
                # TODO High-level modification
                # times = random.randint(1, 2)
                times = 1
                new_count = 0
                while new_count < capacity:


                    # No pattern
                    if args.train_err_mode == 'auto':
                        new_word, action = misspelling_generator_wrapper(word, pure_letters, times, 'none')
                    elif args.train_err_mode == 'byrule':
                        new_word, action = misspelling_generator_wrapper(word, pure_letters, times, langCode)

                    distance = get_distance(new_word, word)

                    if distance > 0:
                        word_similarity[get_distance(new_word, word)].append(new_word)
                        new_count += 1

                words_all_fake_misspl.append(word)
                finish += 1

            words_similarity[word] = word_similarity
            seed_stack.remove(word)

        print("Get edit distance {}s/word".format(round((time.time() - calDist_time)/(60*len(seed_words)), 3)))
        print("Artificial words: {}/{}".format(len(words_all_fake_misspl), args.seed_words))

        # generate batches
        artf_ratio = args.shot_neg_ratio
        if args.shot_neg_ratio <= 0:
            artf_ratio = 0.12
        test_ratio = 0.2
        total_num = args.shot_sample
        data = []
        # fake one to train, true one to test
        for word in seed_words:
            sample_volume = 0
            break_volume = 0
            try:
                distances = [x for x in words_similarity[word].keys() if x > 0]
            except:
                continue
            number_of_dist_type = [len(x) for x in words_similarity[word].values()]
            most_distances_index = number_of_dist_type.index(max(number_of_dist_type))
            # avoid the most edit distance group
            distance_cut = [x for x in distances[:most_distances_index-1] if x <= 2]
            mispl_group = list(itertools.chain.from_iterable([words_similarity[word][i] for i in distance_cut]))

            while sample_volume < total_num and break_volume < 2*total_num:
                break_volume += 1
                if random.randint(1, total_num) / total_num < artf_ratio:
                    # generate fake samples
                    artif_word, action = misspelling_generator_wrapper(word, pure_letters, 1)
                    try:
                        data.append([[v2i['<SOS>']] + [v2i[x] for x in artif_word] + [v2i['<EOS>']], 1, [v2i['<SOS>']] + [v2i[x] for x in word] + [v2i['<EOS>']]])
                    except:
                        sample_volume -= 1

                else:
                    # generate correct samples
                    try:
                        w = [v2i['<SOS>']] + [v2i[x] for x in word.replace("“","").replace("”","")] + [v2i['<EOS>']]
                        data.append([w, 0, w])
                    except:
                        sample_volume -= 1
                sample_volume += 1

        random.shuffle(data)
        train_batches = data

        if args.test_mode == 'fake':
            test_split = int(test_ratio * len(data))
            dev_batches = data[:test_split]

        # Use hash function to select words to form dev set
        if args.test_mode == 'real':
            dev_batches = []
            edit_distances = []
            test_neg_ratio = args.test_neg_ratio
            test_split = int(test_ratio * len(data))
            print("Neg ratio ", test_neg_ratio)
            for i in range(test_split):
                w = random.choice(seed_words)
                f = random.choice(dev_word_distribution[w])
                choose = float(random.randint(1, total_num)) / total_num < test_neg_ratio
                if choose:
                    edit_distances.append(f[1])
                    dev_batches.append([[v2i[x] for x in f[0]], 1, [v2i[x] for x in w]])
                else:
                    dev_batches.append([[v2i[x] for x in w], 0, [v2i[x] for x in w]])

            print('Dev Average Edit distance: ', np.mean(edit_distances))

        if args.test_mode == "unseen":
            dev_batches = [x for x in dev_batches if ''.join([i2v[y] for y in x[2]]) not in seed_words]

        random.seed(args.seed)

    if args.pattern == "normal":
        if args.data == 'en':
            if args.test_num > 0:
                dev_batches = random.sample(dev_batches, args.test_num)

            total_num = args.seed_words*args.shot_sample
            correct_samples = [x for x in train if x[1]==0]
            misspell_samples = [x for x in train if x[1]==1]
            print('misspelled #: ', len(misspell_samples))
            if len(misspell_samples) < int(args.shot_neg_ratio*min(len(train), total_num)):
                gap = int(args.shot_neg_ratio*min(len(train), total_num)) - len(misspell_samples)
                with open("./files/toefl_word_pairs_mispell_clean.txt", 'r') as fp:
                    augment = fp.readlines()
                    augment = [x.strip('\n').strip('\r') for x in augment]
                    negative_augment_sample = random.sample(augment, gap)
                    negative_augment = []
                    for sample in negative_augment_sample:
                        sample = sample.split()
                        try:
                            negative_augment.append([[v2i[x] for x in sample[0]], 1, [v2i[x] for x in sample[2]]])
                        except:
                            pass
                fp.close()
                misspell_samples += negative_augment

            misspell_num = int(args.shot_neg_ratio*min(len(train), total_num))
            train_batches = misspell_samples + random.sample(correct_samples, min(total_num - misspell_num, len(correct_samples)))
            random.shuffle(train_batches)

    if args.pattern == "mixed":
        # mix the true training data and the artificial training data
        total_num = args.seed_words * args.shot_sample
        correct_samples = [x for x in train if x[1] == 0]
        misspell_samples = [x for x in train if x[1] == 1]
        print('misspelled #: ', len(misspell_samples))
        if len(misspell_samples) < int(args.shot_neg_ratio * min(len(train), total_num)):
            gap = int(args.shot_neg_ratio * min(len(train), total_num)) - len(misspell_samples)
            with open("./files/toefl_word_pairs_mispell_clean.txt", 'r') as fp:
                augment = fp.readlines()
                augment = [x.strip('\n').strip('\r') for x in augment]
                negative_augment_sample = random.sample(augment, gap)
                negative_augment = []
                for sample in negative_augment_sample:
                    sample = sample.split()
                    negative_augment.append([[v2i[x] for x in sample[0]], 1, [v2i[x] for x in sample[2]]])
            fp.close()
            misspell_samples += negative_augment

        misspell_num = int(args.shot_neg_ratio * min(len(train), total_num))
        train_batches = misspell_samples + random.sample(correct_samples,
                                                         min(total_num - misspell_num, len(correct_samples)))



        capacity = min(2 * args.seed_words, 50)
        train_dict = set([''.join([i2v[y] for y in x[0]]) for x in train_batches] +
                         [''.join([i2v[y] for y in x[2]]) for x in train_batches])
        dev_distri_num = len(train_dict)
        seed_words = [random.choice(list(train_dict)) for _ in
                      range(min(2 * args.seed_words, dev_distri_num))]
        seed_words = [x for x in seed_words if len(x) >= 3]
        seed_words_selected = []
        words_similarity = {}
        calDist_time = time.time()
        seed_stack = seed_words.copy()
        finish = 0
        while finish < args.seed_words:
            word = seed_stack[-1]
            word_similarity = defaultdict(list)
            count = 0
            for w in train_words_set:
                distance = get_distance(w, word)
                if distance > 0:
                    word_similarity[distance].append(w)
                    count += 1
                    if count > capacity:
                        break
            if len(word_similarity) > 0:
                finish += 1
                seed_words_selected.append(word)
            else:
                # No match in the vocabulary, using fake edition to fill up
                times = 1
                new_count = 0
                while new_count < capacity:
                    new_word = misspelling_generator_wrapper(word, pure_letters, times)
                    distance = get_distance(new_word, word)

                    if distance > 0:
                        word_similarity[get_distance(new_word, word)].append(new_word)
                        new_count += 1

                words_all_fake_misspl.append(word)
                finish += 1

            words_similarity[word] = word_similarity
            seed_stack.remove(word)
        print("Get edit distance {}s/word".format(round((time.time() - calDist_time) / (60 * len(seed_words)), 3)))
        print("Artificial words: {}/{}".format(len(words_all_fake_misspl), args.seed_words))

        # generate batches
        artf_ratio = args.shot_neg_ratio
        if args.shot_neg_ratio <= 0:
            artf_ratio = 0.12
        test_ratio = 0.2
        shot_num = args.shot_sample
        data = []
        # fake one to train, true one to test
        for word in seed_words:
            try:
                distances = [x for x in words_similarity[word].keys() if x > 0]
            except:
                continue
            number_of_dist_type = [len(x) for x in words_similarity[word].values()]
            most_distances_index = number_of_dist_type.index(max(number_of_dist_type))
            # avoid the most edit distance group
            distance_cut = distances[:most_distances_index - 1]
            mispl_group = list(itertools.chain.from_iterable([words_similarity[word][i] for i in distance_cut]))

            for i in range(shot_num):
                try:
                    if random.randint(1, shot_num) / shot_num < artf_ratio:
                        # generate fake samples
                        fake_word = random.choice(mispl_group)
                        data.append([[v2i[x] for x in fake_word], 1, [v2i[x] for x in word]])

                    else:
                        # generate correct samples
                        w = [v2i[x] for x in word]
                        data.append([w, 0, w])
                except:
                    i -= 1
                    pass

        train_batches += data
        random.shuffle(train_batches)
        if len(train_batches) == 0 or len([x for x in train_batches if x[1]==1]) == 0:
            pdb.set_trace()



    # 0 means use all the samples
    if args.test_num > 0:
        if args.test_neg_ratio > 0:
            negative_part = [x for x in dev_batches if x[1] == 1]
            positive_part = [x for x in dev_batches if x[1] == 0]
            negative_num = len(negative_part)
            total_num = int(min(negative_num / args.test_neg_ratio,
                            (len(dev_batches) - negative_num) / (1 - args.test_neg_ratio)))
            neg_part_num = int(total_num * args.test_neg_ratio)
            dev_batches = random.sample(negative_part, neg_part_num) + \
                          random.sample(positive_part, total_num - neg_part_num)
            random.shuffle(dev_batches)
        else:
            dev_batches = random.sample(dev_batches, args.test_num)
    dev_negative = [x for x in dev_batches if x[1]==1]
    dev_num = len(dev_batches)
    origin_dev_batches = dev_batches
    print('Dev #: {} negative: {}({}) Time {}s'.format(dev_num, len(dev_negative), round(len(dev_negative) / dev_num, 3),
                                                       round(time.time() - start_time, 3)))

    # Count Word Frequency
    dev_words = ' '.join(target_dev + source_dev).split()
    dev_words_num = len(dev_words)
    for x in train_words:
        word_freq_train[x] += 1 / train_words_num

    for x in dev_words:
        word_freq_dev[x] += 1 / dev_words_num

    if args.data == 'ru':
        print("Train: {} Dev: {}".format(len(source_train), len(source_dev)))
    if args.data == 'en':
        print("Train: {} neg {} ({}) Dev: {} neg {}".format(len(train_batches), len([x for x in train_batches if x[1]==1]),
                                            round(len([x for x in train_batches if x[1]==1]) / len(train_batches), 3),
                                            len(dev_batches), len([x for x in dev_batches if x[1]==1])))

    mistake_hit = defaultdict(int) # which are the correct words get corrected from the mistake form
    correct_hit = defaultdict(int)
    fp_hit = defaultdict(int)
    metrics = {'precision':[], 'recall':[], 'f1':[], 'eval_accuracy':[], 'eval_negative_accuracy':[], 'abs_identifier':[]}
    STOP_WARNING = 0
    stop_tolerance = 7
    loss = []
    train_acc = []
    TN_dict = defaultdict(int)
    TP_dict = defaultdict(int)
    inconfident_number = []
    yield_batches = False
    activation = args.activation
    for epoch in range(NEPOCH):
        action_collecter.newCollector()
        is_best = False
        start = time.time()
        if args.pattern not in ["random_zero_shot", "normal", "mixed"]:
            train_batches = list(get_batches(source_train, target_train, batch_size, v2i, padding_mode='max',
                                             level='binary', to_yield=yield_batches))
        else:
            random.shuffle(train_batches)
        train_num = len(train_batches)
        if not yield_batches:
            num_of_batches = int(len(train_batches) / batch_size)
        train_negative = [x for x in train_batches if x[1] == 1]
        train_negative_word_freq = defaultdict(int)
        for w in train_negative:
            word = ''.join([i2v[x] for x in w[0] if x != v2i['<PAD>']])
            train_negative_word_freq[word] += 1
        encoder.train()
        random.shuffle(train_batches)
        train_predict = 0
        train_correct = 0
        train_correct_negative = 0
        test_correct = 0
        test_correct_negative = 0
        test_correct_negative_sample = []
        test_FP_sample = []
        train_loss = 0
        process = 0
        zero_pass = 0
        TP = 0
        FP = 0
        FN = 0

        if args.model == 'rnn':
            for i in range(num_of_batches):
                batch = train_batches[i*batch_size: (i+1)*batch_size]
                input_lengths = [len(x[0]) for x in batch]
                max_input_length = max(input_lengths) + 2
                max_target_length = max([len(x[2]) for x in batch]) + 2
                batch_copy = batch.copy()
                batch_new = []
                for j in range(len(batch)):
                    if v2i['<EOS>'] not in batch[j][0]:
                        batch[j][0] += [v2i['<EOS>']]
                    if v2i['<EOS>'] not in batch[j][2]:
                        batch[j][2] += [v2i['<EOS>']]
                    if len(batch[j][0]) < max_input_length:
                        source_w = batch[j][0] + [v2i['<PAD>']] * (max_input_length - len(batch[j][0]))
                    else:
                        source_w = batch[j][0][:max_input_length-1] + [v2i['<EOS>']]

                    if len(batch[j][2]) < max_target_length:
                        target_w = batch[j][2] + [v2i['<PAD>']] * (max_target_length - len(batch[j][2]))
                    else:
                        target_w = batch[j][2][:max_target_length-1] + [v2i['<EOS>']]

                    batch_new.append([source_w, batch[j][1], target_w])

                batch = batch_new.copy()
                train_tensor = Variable(torch.tensor([x[0] for x in batch]).type(Type))
                target_tensor = Variable(torch.tensor([x[2] for x in batch]).type(Type))
                optimizer.zero_grad()
                tags_train = [x[1] for x in batch]

                # encoder_outputs of [batch, max_seq_len, hidden_size]
                # encoder_hidden of [2*layer, max_seq_len, hidden_size]
                encoder_outputs, encoder_hidden = encoder(train_tensor, input_lengths, padded=False)
                encoder_last_outputs = encoder_outputs[:, -1, :]

                # decoder_input = Variable(torch.LongTensor([v2i['<UNK>']] * batch_size))
                # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
                scores = encoder.projection(encoder_last_outputs)
                predicts = scores.argmax(dim=1).cpu().numpy()
                scores_prob = softmax(scores.detach().numpy(), axis=1)

                # Collect examples hard to train
                inconfident_indexes_high = np.where(scores_prob[:, 1] > 0.45)[0].tolist()
                inconfident_indexes_low = np.where(scores_prob[:, 1] < 0.55)[0].tolist()
                inconfident_indexes = [x for x in inconfident_indexes_low if x in inconfident_indexes_high]
                inconfident_number.append(len(inconfident_indexes))
                hard_samples[epoch] = hard_samples[epoch] + [batch[x] for x in inconfident_indexes]

                TN_indexes = set(np.where(predicts==0)[0]) & set(np.where(np.array(tags_train)==1)[0])
                TP_indexes = np.where((predicts==tags_train)==True)[0]

                for TN_index in TN_indexes:
                    TN_dict[''.join([i2v[x] for x in batch[TN_index][0] if x not in [v2i['<PAD>'], v2i['<SOS>'], v2i['<EOS>']]])] += 1

                for TP_index in TP_indexes:
                    TP_dict[''.join([i2v[x] for x in batch[TP_index][0] if x not in [v2i['<PAD>'], v2i['<SOS>'], v2i['<EOS>']]])] += 1

                train_correct += sum(predicts == tags_train)
                train_correct_negative += sum([x*y for x, y in zip(predicts, tags_train)])
                my_loss = criterion(scores, torch.tensor(tags_train).type(Type))
                train_loss += my_loss.item()
                my_loss.backward()
                optimizer.step()

        if args.model == 'cnn':
            for train_batch, tag_batch, target_batch in train_batches:
                if len(train_batch) == 0:
                    zero_pass += 1
                    continue

                if len(train_batch) < WIN_SIZE_MAX:
                    train_batch += [v2i['<PAD>']] * (WIN_SIZE_MAX - len(train_batch))

                train_tensor = torch.tensor(train_batch).type(Type)
                target_tensor = torch.tensor([target_batch]).type(Type)

                scores = model(train_tensor)  # batch * 1 * ntags
                predicts = scores.argmax().cpu().numpy()
                train_predict += predicts.tolist()
                train_correct += (predicts == tag_batch)

                # record the predicted mispelled ones
                train_correct_negative += (predicts and tag_batch)
                my_loss = criterion(scores.unsqueeze(0), torch.tensor(tag_batch).type(Type).unsqueeze(0))
                train_loss += my_loss.item()
                process += 1

                if process % batch_size == 0:
                    # Do back-prop
                    for optimizer in optimizers.values():
                        optimizer.zero_grad()
                    my_loss.backward()
                    for optimizer in optimizers.values():
                        optimizer.step()
                    my_loss = 0

        loss.append(train_loss / len(train_batches))
        train_acc.append(train_correct / len(train_batches))

        print("\niter %r: train loss/sent=%.4f, acc=%.4f, negative acc=%.4f time=%.2fs" % \
              (epoch, train_loss / train_num, train_correct / train_num,
               train_correct_negative / len(train_negative), round(time.time() - start, 3)))

        if np.isnan(train_correct_negative / len(train_negative)):
            # TODO analyze  this error
            train_negative.append(0)


        # Test
        for model in models.values():
            model.eval()

        for i, [dev_batch, tag_dev, dev_batch_gold, action] in enumerate(dev_batches):
            if len(dev_batch) == 0:
                continue

            if epoch == 0:
                dev_batch = [v2i['<SOS>']] + dev_batch + [v2i['<EOS>']]
                dev_batch_gold = [v2i['<SOS>']] + dev_batch_gold + [v2i['<EOS>']]
            dev_tensor = torch.tensor(dev_batch).type(Type)

            if args.model == 'rnn':
                encoder_outputs, encoder_hidden = encoder(dev_tensor.unsqueeze(0), len(dev_batch),
                                                          activation=activation, padded=False, hidden=None)
                encoder_last_outputs = encoder_outputs[:, -1, :]
                # decoder_input = Variable(torch.LongTensor([v2i['<UNK>']] * batch_size))
                # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
                scores = encoder.dense(encoder_last_outputs)
                predicts = scores.argmax(dim=1).cpu().numpy()

            if args.model == 'cnn':
                scores = model(dev_tensor)  # batch * 1 * ntags
                predicts = scores.argmax().cpu().numpy()

            test_correct += (predicts == tag_dev)
            FP += (predicts == 1)
            if predicts == tag_dev:
                correct_hit[''.join([i2v[x] for x in dev_batch_gold]).rstrip()] += 1

            # Use to record correct spelling misclassified
            if predicts - tag_dev == 1:
                word = ''.join([i2v[x] for x in dev_batch]).rstrip()
                test_FP_sample.append((word, round(word_freq_train[word], 6)))
                fp_hit[word] += 1

        for dev_batch, tag_dev, dev_batch_target, action in dev_negative:
            if len(dev_batch) == 0:
                continue

            if dev_batch[-1] != v2i['<EOS>']:
                dev_batch = [v2i['<SOS>']] + dev_batch + [v2i['<EOS>']]

            dev_w = ''.join([i2v[x] for x in dev_batch]).rstrip('<EOS>').lstrip('<SOS>')
            dev_target_w = ''.join([i2v[x] for x in dev_batch_target]).rstrip('<EOS>').lstrip('<SOS>')
            dev_tensor = torch.tensor(dev_batch).type(Type)
            target_tensor = torch.tensor([dev_batch]).type(Type)
            if args.model == 'cnn':
                scores = model(dev_tensor)  # batch * 1 * ntags
                predicts = scores.argmax().cpu().numpy()
            if args.model == 'rnn':
                encoder_outputs, encoder_hidden = encoder(dev_tensor.unsqueeze(0), len(dev_batch), activation=activation,\
                                                            padded=False, hidden=None)
                encoder_last_outputs = encoder_outputs[:, -1, :]
                scores = encoder.dense(encoder_last_outputs)
                predicts = scores.argmax(dim=1).cpu().numpy()

            test_correct_negative += (predicts == tag_dev)
            if predicts == tag_dev:
                test_correct_negative_sample.append((''.join([i2v[x] for x in dev_batch]).rstrip(), \
                                                     round(word_freq_train[''.join([i2v[x] for x in dev_batch])],6),
                                                     ''.join([i2v[x] for x in dev_batch_target]).rstrip(),
                                                     round(word_freq_train[''.join([i2v[x] for x in dev_batch_target])], 6)))
                mistake_hit[dev_w] += 1
                action_collecter.step_collector['succeed'][action] += 1
                if len(KB_correction[dev_w]) == 0:
                    KB_correction[dev_w] = dev_target_w
                TP += 1

                # count the occurrence of the misspelled words
                new_misp_error_dict[''.join([i2v[x] for x in dev_batch]).rstrip()] += 1

            else:
                action_collecter.step_collector['fail'][action] += 1

        FN = test_correct - TP

        # Take the misspelled as positive
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2*TP / (2*TP + FP + FN)
        metrics['precision'].append(Precision.tolist())
        metrics['recall'].append(Recall.tolist())
        metrics['f1'].append(F1.tolist())
        metrics['eval_accuracy'].append((test_correct / dev_num).tolist())
        metrics['eval_negative_accuracy'].append((test_correct_negative / len(dev_negative)).tolist())
        metrics['abs_identifier'].append(len(set(test_correct_negative_sample)))

        # if test_correct_negative / len(dev_negative) >= best_model:
        if F1.tolist()[0] >= best_f1:
            best_model = test_correct_negative / len(dev_negative)
            best_f1 = max(best_f1, F1)
            is_best = True

            # save the parameter
            best_model_parameter['model'] = copy.deepcopy(encoder.state_dict())
            best_model_parameter['optimizer'] = copy.deepcopy(optimizer.state_dict())


            if (test_correct / dev_num).tolist()[0] > 0.3:
                best_model_metrics['train_acc'] = train_correct / train_num
                best_model_metrics['eval_acc'] = (test_correct / dev_num).tolist()[0]
                best_model_metrics['best_epoch'] = epoch
                best_model_metrics['neg_acc'] = (test_correct_negative / len(dev_negative)).tolist()[0]
                best_model_metrics['precision'] = Precision.tolist()[0]
                best_model_metrics['recall'] = Recall.tolist()[0]
                best_model_metrics['F1'] = F1.tolist()[0]

        else:
            if epoch > 20:
                if args.early_stop == 'immediate':
                    print('Early stop at epoch', epoch)
                    break

                if args.early_stop == 'no':
                    pass

                if args.early_stop == 'tolerate':
                    STOP_WARNING += 1
                    if STOP_WARNING == stop_tolerance:
                        break

        if len(test_FP_sample) > 0:
            test_FP_sample.sort(key=lambda x: x[1])
            test_FP_sample = test_FP_sample[::-1]

        if len(test_correct_negative_sample) > 0:

            # sort by frequency of error
            test_correct_negative_sample.sort(key=lambda x: x[1])

            # from high freq to low
            test_correct_negative_sample = test_correct_negative_sample[::-1]

        if is_best:
            best_correct = test_correct_negative_sample
            best_wrong = test_FP_sample

        action_collecter.merge()
        print('Correct ones ', test_correct_negative_sample)
        if len(set(test_FP_sample)) > 100:
            print('\nWrong predicted ', list(set(test_FP_sample))[:100])
        else:
            print('\nWrong predicted ', list(set(test_FP_sample)))
        print('\naction seq succeed: ', sorted(action_collecter.step_collector['succeed'].items(), key=operator.itemgetter(1))[::-1])
        print('\naction seq fail: ', sorted(action_collecter.step_collector['fail'].items(), key=operator.itemgetter(1))[::-1])
        print('Eval acc=%.4f, negative correct acc=%.4f best=%.4f abs#=%d(%d) #fp=%d(%d)'%
              (test_correct / dev_num, test_correct_negative / len(dev_negative), best_model,
               test_correct_negative, len(set(test_correct_negative_sample)), len(test_FP_sample), len(set(test_FP_sample))))
        print('Precision=%.4f Recall=%.4f F1=%.4f best F1=%.4f' % (Precision, Recall, F1, best_f1))


    metrics['loss'] = loss
    metrics['train_accuracy'] = train_acc
    correct_hit_cp = correct_hit.copy()
    correct_hit_cp = sorted(correct_hit.items(), key=lambda x: x[1] / (epoch + 1))
    mistake_hit = sorted(mistake_hit.items(), key=lambda x: x[1] / (epoch + 1))

    correct_summary = [(key[0], key[1], key[1] / max(train_negative_word_freq[key[0].rstrip()], 1), word_freq_train[key[0].rstrip()]) for key in correct_hit_cp]
    mistake_summary = [(key[0], key[1], key[1] / max(train_negative_word_freq[key[0].rstrip()], 1),
                        word_freq_train[key[0].replace(' ', '')]) for key in mistake_hit]

    if type(best_model) == np.ndarray:
        best_model = best_model[0]
    if args.save_result and best_f1 > 0.1:
        # Record hyperparameters
        metrics['pattern'] = args.pattern
        metrics['shot_sample'] = args.shot_sample
        artf_ratio = args.shot_neg_ratio
        if args.shot_neg_ratio <= 0:
            artf_ratio = 0.12
        metrics['shot_negative_ratio'] = artf_ratio
        metrics['test_neg_ratio'] = args.test_neg_ratio
        metrics['bidirection'] = bidirection
        metrics['test_mode'] = args.test_mode
        metrics['mode'] = args.mode
        metrics['layer'] = args.layers
        metrics['data'] = args.data
        metrics['model'] = args.model
        metrics['embedding'] = args.pattern
        metrics['hidden_size'] = hidden_size
        metrics['embedding_size'] = embedding_size
        metrics['batch_method'] = args.mode
        metrics['batch'] = args.batch
        metrics['filter'] = args.filter
        metrics['dropout'] = args.dropout
        metrics['ratio'] = args.ratio
        metrics['weight_norm'] = args.weight_norm
        metrics['seed_words'] = args.seed_words
        metrics['train_err_mode'] = args.train_err_mode
        metrics['test_err_mode'] = args.err_mode
        metrics['mask'] = args.mask

        metrics['weight_decay'] = args.weight_decay
        metrics['optimizer'] = args.opt
        metrics['random_seed'] = args.seed
        metrics['fp_sample'] = fp_hit
        metrics['epoch'] = epoch
        metrics['best_correct'] = best_correct
        metrics['best_wrong'] = best_wrong

        metrics['best_f1'] = max(F1.tolist())
        metrics['best_eval_negative_accuracy'] = best_model
        metrics['best_eval_accuracy'] = max(metrics['eval_accuracy'])
        metrics['actions'] = action_collecter.collector
        summary_saver(metrics)
        print('Data saved ...')

    print('#Dev ', len(dev_batches))
    print('------------------------------------------------------------------------------------------------')

    print('Finish')
    print('best model ', best_model_metrics)

    tune_summary = {'loss':[], 'accuracy':[], 'volume':[], 'trigger': ['None']}
    best_epoch = best_model_metrics['best_epoch']
    tune_epoch = 20

    # Statistics of the hard trained example
    batch_hard = hard_samples[best_epoch]
    batch_hard_pad = []
    correct_hard = 0
    my_loss_hard = 0
    tune_max_length = 16
    encoder.eval()
    for sample in batch_hard:
        if len(sample[0]) < tune_max_length:
            batch_hard_pad.append([sample[0]+[v2i['<PAD>']] * (tune_max_length - len(sample[0])), sample[1], sample[2]])
        else:
            batch_hard_pad.append([sample[0][:tune_max_length-1] + [v2i['<EOS>']], sample[1], sample[2]])
        train_tensor_hard = Variable(torch.tensor(sample[0]).type(Type))
        target_tensor_hard = Variable(torch.tensor(sample[2]).type(Type))
        input_length_hard = len([x for x in sample[0] if x != 0])
        encoder.eval()
        optimizer.zero_grad()
        tags_train_hard = sample[1]
        encoder_outputs, encoder_hidden = encoder(train_tensor_hard.unsqueeze(0), input_length_hard, padded=False)
        encoder_last_outputs = encoder_outputs[:, -1, :]
        scores_hard = encoder.projection(encoder_last_outputs)
        predicts_hard = scores.argmax(dim=1).cpu().numpy()

    correct_hard += (predicts_hard == tags_train_hard)
    train_correct_negative += int(predicts_hard * tags_train_hard)
    my_loss_hard += criterion(scores_hard, torch.tensor([tags_train_hard]).type(Type))
    print("Finish tuning of hard example of epoch {} loss/sample: {} acc: {}".format(best_epoch,
                                                                                     round(my_loss.item()/len(batch_hard), 5),
                                                                                     correct_hard[0] / len(batch_hard)))
    tune_summary['loss'].append(my_loss/len(batch_hard))
    tune_summary['accuracy'].append(correct_hard[0] / len(batch_hard))
    tune_summary['volume'].append(len(batch_hard))

    if args.data != "en":
        result_decision = {}
        for i, [dev_batch, tag_dev, dev_batch_gold, action] in enumerate(dev_batches):
            if len(dev_batch) == 0:
                continue

            dev_word = ''.join([i2v[x] for x in dev_batch]).lstrip('<SOS>').rstrip('<EOS>')
            dev_word_gold = ''.join([i2v[x] for x in dev_batch_gold]).lstrip('<SOS>').rstrip('<EOS>')
            dev_tensor = torch.tensor(dev_batch).type(Type)

            if args.model == 'rnn':
                encoder_outputs, encoder_hidden = encoder(dev_tensor.unsqueeze(0), len(dev_batch),
                                                          activation=activation, padded=False, hidden=None)
                encoder_last_outputs = encoder_outputs[:, -1, :]
                scores = encoder.dense(encoder_last_outputs)
                predicts = scores.argmax(dim=1).cpu().numpy()
                result_decision[dev_word] = {}
                result_decision[dev_word]['decision'] = predicts[0]
                result_decision[dev_word]['golden'] = dev_word_gold
                result_decision[dev_word]['dict'] = TN_dict.get(dev_word, 0) + TP_dict.get(dev_word, 0)
                result_decision[dev_word]['dev_hit'] = correct_hit.get(dev_word, 0)

        try:
            directory = './online_test/{}/'.format(data_dict[args.data])
            sub_dir = test_file.split('/')[0]
            if not os.path.isdir(directory + '/' + sub_dir):
                os.mkdir(directory + '/' + sub_dir)

            with open(directory + test_file.rstrip('.txt') + '_f1_{}.pkl'.format(int(round(best_f1[0], 3)*1000)), 'wb') as fp:
                pickle.dump(result_decision, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

            print("------------------------- Test file summary ---------------------------------------------------")
            result_analysis(result_decision)
        except Exception as ex:
            print(ex)
            pdb.set_trace()
        print("Save evaluation sample of " + test_file)


    print('\nTest, type end to terminate')

    # load the parameter of the best model
    if len(best_model_parameter['model']) > 0:
        # the best model is saved
        encoder.load_state_dict(best_model_parameter['model'])
        optimizer.load_state_dict(best_model_parameter['optimizer'])

    word_hiddens = defaultdict(dict)
    decisions = {0: 'correct', 1: 'misspelled'}
    while True:
        input_string = input("\nTest a word: ")
        if input_string == "end":
            break

        evaluate_sample = [v2i['<SOS>']] + [v2i[x] for x in input_string] + [v2i['<EOS>']]

        if args.model == 'rnn':
            sample_scores, sample_predicts, hiddens = evaluate(evaluate_sample, v2i, args.model, encoder,
                                                                args.activation, return_hidden=True)
        word_predict = ''.join([i2v[x] for x in predicts])
        print("{} freq: {}%".format(input_string, round(word_freq_train[input_string]*100, 5)))
        sample_scores_show = sample_scores.detach().numpy()
        sample_prob = softmax(sample_scores_show, axis=1)
        prob_chain = softmax(encoder.dense(hiddens).detach().numpy(), axis=2)
        word_hiddens[input_string] = {'hiddens': hiddens, 'likelihood': sample_scores_show,
                                      'probs': prob_chain[0][:, 0].tolist()}
        print("correct {}({}) incorrect {}({}) decision: {} findInKB: {}".format(round(sample_scores_show[0][0], 3),
                                                                  round(sample_prob[0][0], 3),
                                                                  round(sample_scores_show[0][1], 3),
                                                                  round(sample_prob[0][1], 3),
                                                                  decisions[sample_predicts.tolist()[0]],
                                                                  KB_correction[input_string]))

        print("prob chain of correct: ", [round(x, 4) for x in prob_chain[0][:, 0].tolist()])
        print("hit in training: ", TP_dict[input_string], " prediction: ", correct_hit[input_string])
        response = input("Enter 1 for correct decision, 0 for wrong decision: ")
        try:
            response = int(response)
            if not response:
                print("New word to correct: ", input_string)

            if not response^sample_predicts.tolist()[0]:
                #  model: correct/misspelled human: model is wrong
                if not TP_dict[input_string]:
                    KB_update = input("Update {} to list? (1 for Yes, 0 for Pass): ".format(input_string))
                    try:
                        KB_update = int(KB_update)
                        if KB_update:
                            TP_dict[input_string] += 1
                            KB_correction[input_string] = input_string
                    except:
                        pass

            if response == 0 and sample_predicts.tolist()[0]:

                # model: wrong human: correct spelled
                batch_hard_pad, tune_summary = tuning_model(encoder, optimizer, v2i,
                                                            (input_string, None), 0, batch_hard_pad, tune_summary,
                                                            tune_epoch=tune_epoch, Type=Type, criterion=criterion)

            # model: correct human:misspelling
            if not (response + sample_predicts.tolist()[0]):
                # retrieve N words from dictionary using edit distance
                N = 10
                sample_word_similarities = defaultdict(set)
                count = 0
                for word in TP_dict.keys():
                    try:
                        sample_distance = get_distance(word, input_string)
                        if sample_distance > 0 and sample_distance <= 7:
                            sample_word_similarities[sample_distance].add(word)
                            count += int(sample_distance == 1)
                            if count > N:
                                break
                    except:
                        pdb.set_trace()


                if count:
                    print("For the following {} retrieved words, enter 1 if correct spelling and 0 o/w:".format(count))

                    sorted_distance = dict(sorted(sample_word_similarities.items(), key=lambda x: x[0]))
                    select = []

                    # Select at most top N words in the list
                    for distance, word_list in sorted_distance.items():
                        for word in word_list:
                            select.append(word)
                            if len(select) == N:
                                break

                    rank = 1
                    while True:
                        retrieve_response = input("retrieval word # {}: {} correct spelling?(1 for Yes, 0 for No, exit for exit): ".format(rank, select[rank-1]))
                        if retrieve_response == 'exit':
                            break
                        try:
                            retrieve_response = int(retrieve_response)
                            if retrieve_response:
                                KB_correction[input_string] = select[rank-1]
                                print("New word added: {} -> {}".format(input_string, select[rank-1]))

                                # Still Tuning the model
                                tuning = input("Tune the model for {}? (1 for Yes and 0 for Pass): ".format(input_string))
                                try:
                                    tuning = int(tuning)
                                    if tuning:
                                        batch_hard_pad, tune_summary = tuning_model(encoder, optimizer, v2i,
                                                                                    (input_string, select[rank-1]), 0, batch_hard_pad,
                                                                                    tune_summary, tune_epoch=tune_epoch, Type=Type,
                                                                                    criterion=criterion)
                                        print("Finish Tuning of {}".format(input_string))
                                except:
                                    pass

                                break
                            rank += 1
                            if rank > len(select):
                                print("No words retrieved")
                                break
                        except:
                            pass
                else:
                    # No words retrieved from the vocabulary
                    print("no word retrieved")
                    pass

        except:
            pass

