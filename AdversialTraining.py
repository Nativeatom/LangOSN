import os
import string
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from RuPreprocess import RNN, evaluate,\
                        get_distance, \
                        load_model, save_model, \
                        word_to_index, position_cal
from config import *
from OnlineSpellingServer import get_model, \
                                 build_vocab, get_optimizer, \
                                 get_loss_criterion, build_minion_group, \
                                 token2idxs
from training import evaluate_, evaluate_wrapper
from Useful_Function import softmax, character_preprocessing, misspelling_generator_wrapper
from collections import defaultdict, Counter
from sklearn.metrics import precision_recall_fscore_support
from evaluation import metrics_cal
import itertools
import pdb

# Parameters
loss_criterion = "CrossEntropy"
criterion = get_loss_criterion(loss_criterion)
activation = "lstm"
decisions = {0: 'correct', 1: 'misspelled'}
Type = torch.LongTensor
times = 1
lr = 0.0001
lr_decay = 1
minion_group_size = 15 # 15
minion_group_neg_ratio = 0.5
tune_epoch = 8
use_batch = True
batch_size = 15 # 15
weights = [1, 0]
eval_every_iter = 20
hard_sample_stop = False
early_stop = False
# stop_mode = 'oneEpoch'
stop_mode = 'no'
lang = 'English'

# tuning adversial samples and true samples in random order or not
summary = {'loss': [], 'accuracy': [], 'dev_acc': [], 'dev_f1': [],
           'dev_acc_pos': [], 'trigger': ['None']}
tuning_mode = 'random'

def sample_eval(token, vocab2index, model, activation='lstm', Type=torch.LongTensor, show_result=True, return_prob=True):
    evaluate_sample = [vocab2index['<SOS>']] + [vocab2index[x] for x in token] + [vocab2index['<EOS>']]
    sample_scores, sample_predicts, hiddens = evaluate(evaluate_sample, vocab2index, 'rnn', model,
                                                       activation, Type=Type, return_hidden=True)
    sample_scores_show = sample_scores.detach().numpy()
    sample_prob = softmax(sample_scores_show, axis=1)
    result_model = decisions[sample_predicts.tolist()[0]]
    if show_result:
        print('{} correct in {}'.format(token, round(sample_prob[0][0], 2)))

    if return_prob:
        return sample_prob, sample_predicts

def tuning_model(model, optimizer, word2index, new_word_set, label, minion_group, summary, tune_epoch, Type, criterion,
                 number=20, batch=False, early_stop=False, batch_size=10,
                 protect_epoch=0, weights=[0.5, 0.5], eval_every_iter=50,
                 hard_sample_stop=False, dev_set=(), model_save=False, lr_decay=1, acc_gap=0.01, early_break=True,
                 stop_strategy='early_stop', log=False, return_model=False, show_model_status=False):

    # TODO rotate the member in the group
    max_length = 16
    print("Tuning model...")
    if log:
        logger.info("Tuning model...")
    model.train()
    new_word, new_word_correct = new_word_set
    minion_group_size = sum([len(x) for x in minion_group])
    if type(new_word) == list:
        data_size = minion_group_size + len(new_word)
    elif type(new_word) == str:
        data_size = minion_group_size + 1 # for new_word
    else:
        data_size = minion_group_size
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

    if stop_strategy == 'oneEpoch':
        tune_epoch = 1

    lambda_mspl, lambda_pos = [x/sum(weights) for x in weights]

    if not batch:
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

        # target_tensor_tune = Variable(torch.tensor([x[2] for x in batch_nove]).type(Type))
        my_loss = 0
        for epoch in range(tune_epoch):
            optimizer.zero_grad()
            if show_model_status:
                print("epoch {} model training {} loss: {}".format(epoch, model.training, round(my_loss, 4)))
                sample_eval('bicause', vocab2index, model)

            random.shuffle(batch_nove)
            train_tensor_tune = Variable(torch.tensor([x[0] for x in batch_nove]).type(Type))
            input_length_nove = [len(x[0]) for x in batch_nove]
            tags_train = [x[1] for x in batch_nove]
            pos_train = [x[3] for x in batch_nove]
            encoder_outputs, encoder_hidden = model(train_tensor_tune, input_length_nove, padded=False)
            encoder_last_outputs = encoder_outputs[:, -1, :]
            scores = model.projection(encoder_last_outputs)
            position_scores = model.position_projection(encoder_last_outputs)
            predicts = scores.argmax(dim=1).cpu().numpy()
            position_predicts = position_scores.argmax(dim=1).cpu().numpy()
            my_loss = lambda_mspl * criterion(scores, torch.tensor(tags_train).type(Type)) + \
                      lambda_pos * criterion(position_scores, torch.tensor(pos_train).type(Type))
            my_loss.backward()
            optimizer.step()
            iter += 1

        print("Finish tuning of {}".format(new_word))
        print("avg_loss: {} acc: {} volume: {}".format(round(my_loss.item() / len(batch_nove), 6),
                                                   round(sum(predicts == tags_train) / len(batch_nove), 4),
                                                   len(batch_nove)))
        if log:
            logger.info("Finish tuning of {}".format(new_word))
            logger.info("avg_loss: {} acc: {} volume: {}".format(round(my_loss.item() / len(batch_nove), 6),
                                                           round(sum(predicts == tags_train) / len(batch_nove), 4),
                                                           len(batch_nove)))
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


        if len(dev_set) > 0:
            do_dev = True
            test_words, test_tags, test_pos_tags = dev_set
            test_tensor = Variable(torch.tensor([word_to_index(x, word2index, max_length, pad=True) for x in test_words]).type(Type))

        epoch_loss = 0
        my_loss = torch.tensor(0)
        for epoch in range(tune_epoch):
            if show_model_status:
                print("epoch {} model training {} loss: {}".format(epoch, model.training, round(my_loss.item(), 4)))
                # sample_eval('because', vocab2index, model)
            random.shuffle(batch_nove)
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
                encoder_outputs, encoder_hidden = model(train_tensor, input_lengths, padded=False)
                encoder_last_outputs = encoder_outputs[:, -1, :]

                scores = model.projection(encoder_last_outputs)
                predicts = scores.argmax(dim=1).cpu().numpy()
                scores_prob = softmax(scores.detach().numpy(), axis=1)

                position_scores = model.position_projection(encoder_last_outputs)
                position_predicts = position_scores.argmax(dim=1).cpu().numpy()

                inconfident_indexes_hign = np.where(scores_prob[:, 1] > 0.45)[0].tolist()
                inconfident_indexes_low = np.where(scores_prob[:, 1] < 0.55)[0].tolist()
                inconfident_indexes = [x for x in inconfident_indexes_low if x in inconfident_indexes_hign]
                inconfident_number.append(len(inconfident_indexes))
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
                    encoder_outputs_eval, encoder_hidden_eval = model(test_tensor, len(test_words), padded=False)
                    encoder_last_outputs_eval = encoder_outputs_eval[:, -1, :]

                    # TODO: auxiliary task of autoencoding?
                    # decoder_input = Variable(torch.LongTensor([v2i['<UNK>']] * batch_size))
                    # decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
                    scores_eval = model.projection(encoder_last_outputs_eval)
                    predicts_eval = scores_eval.argmax(dim=1).cpu().numpy()
                    test_acc = sum([x == y for x, y in zip(predicts_eval, test_tags)])

                    position_scores_eval = model.position_projection(encoder_last_outputs_eval)
                    position_predicts_eval = position_scores_eval.argmax(dim=1).cpu().numpy()
                    test_position_acc = sum([x == y for x, y in zip(position_predicts_eval, test_pos_tags)])
                    model.train()

                    test_accuracy, _,  _, f1 = metrics_cal(predicts=predicts_eval, tags=test_tags, detail=False)
                    losses.append(epoch_loss / (div*batch_size))
                    accs.append(acc / (div*batch_size))
                    # test_accs.append(test_acc / len(test_words))
                    test_accs.append(test_accuracy)
                    test_accs_pos.append(test_position_acc / len(test_words))
                    dev_f1s.append(f1)
                    print("#hard samples: ", inconfident_number[-1])
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
                                    summary['langcode'],
                                    len(new_word),
                                    batch_size,
                                    epoch,
                                    iter,
                                    round(test_accuracy, 3),
                                    round(f1, 3),
                                    curr_lr)
                                save_model(model, name)

                            if early_break:
                                break

                    epoch_loss = 0

        print("Finish tuning of {} tokens like {}".format(len(new_word), random.choice(new_word)))
        if log:
            logger.info("Finish tuning of {} tokens like {}".format(len(new_word), random.choice(new_word)))

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



if __name__ == "__main__":
    lang_folder = lang
    langCode = lang2code[lang]
    pure_letters = langCode2alphabet_pure[langCode]
    model_path = './model'
    vocab2index, index2vocab = build_vocab(langCode)
    model = get_model('rnn', len(index2vocab))
    optimizer = get_optimizer(model, name='adam', lr=lr)
    decision_probs = []
    alter_decision_probs = []
    model, optimizer = load_model(model, optimizer, model_path + '/en_50_F1_Acc0.94_F10.48_model.pth.tar',
                       load_optimizer=False, return_model=True)

    if lang_folder == 'eng_TOEFL':
        langcode = 'en'
        lang_path_addition = ''
    elif lang_folder == 'English':
        langcode = lang2code[lang_folder]
        lang_path_addition = '/USCities'
    else:
        langcode = lang2code[lang_folder]
        lang_path_addition = ''

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
                                                                                                    langcode), 'r') as fp:
        test2 = fp.readlines()
        test2 = [x.strip('\n').split() for x in test2]
        test2 = [x for x in test2 if '\u200b' or '°' not in x[0] and len(x) > 0]
    fp.close()

    test_words = [character_preprocessing(x[0]) for x in test]
    test_tags = [int(x[1]) for x in test]

    test2_words = [character_preprocessing(x[0]) for x in test2]
    test2_tags = [int(x[1]) for x in test2]

    if lang_folder not in ['Ainu', 'Griko']:
        # pure correct set
        with open("./files/{}/city{}/high_frequent/{}_most_frequent_200_in_whole_oov_200_pure_correct.txt".format(
                lang_folder, lang_path_addition, langcode), "r") as fp:
            pure_correct = fp.readlines()
            pure_correct = [x.strip('\n').split() for x in pure_correct]
            pure_correct = [x for x in pure_correct if '\u200b' or '°' not in x[0] and len(x) > 0]
        fp.close()
        pure_correct_words = [character_preprocessing(x[0]) for x in pure_correct]
        pure_correct_tags = [int(x[1]) for x in pure_correct]

    try:
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
        print("no dev set")
        dev_words = []
        dev_tags = []
        dev_pos_tags = []

    token_lower = 'because'
    token_lower_alter = 'bicause'
    label_alter = 0

    alter_ratio = 0.2
    alter_num = 10

    alter_list = [misspelling_generator_wrapper(token_lower, pure_letters, times, langCode)[0] for _ in
                  range(int(alter_num * alter_ratio))]
    alter_list += [token_lower] * (alter_num - len(alter_list))

    tuning_mode = 'random'
    batch_hard_pad = []

    if tuning_mode == 'random':
        random.seed(0)
        random.shuffle(alter_list)


    print("--------------------- {} - {} origin -----------------".format(token_lower, token_lower_alter))
    token_prob, token_predict = sample_eval(token_lower, vocab2index, model, activation, Type,
                                            show_result=True, return_prob=True)
    token_prob_alter, token_predict_alter = sample_eval(token_lower_alter, vocab2index, model, activation, Type,
                                                        show_result=True, return_prob=True)
    model_decision = decisions[token_predict.tolist()[0]]
    decision_probs.append(token_prob[0][0])
    alter_decision_probs.append(token_prob_alter[0][0])
    test_summary = evaluate_wrapper(test_words, test_tags, vocab2index, model, Type, activation, title='Test',
                                    show_dist=True, return_prob=True)
    test2_summary = evaluate_wrapper(test2_words, test2_tags, vocab2index, model, Type, activation, title='Test2',
                                     show_dist=True, return_prob=True)

    for token_id, token in enumerate(alter_list):
        print("--------- #{} token {} tuning ---------------------".format(token_id, token))
        minion_group, minion_group_tokens = build_minion_group(lang, vocab2index, token, minion_group_size,
                                                               minion_group_neg_ratio, mode='byrule',
                                                               adding_pos=True)
        batch_hard_pad.append(minion_group)

        model.train()

        if use_batch:
            batch_hard_pad, summary, model = tuning_model(model, optimizer, vocab2index,
                                                          ([token], [token]), label_alter, batch_hard_pad,
                                                          summary, tune_epoch=tune_epoch, Type=Type, criterion=criterion,
                                                          batch=use_batch, batch_size=batch_size, protect_epoch=0,
                                                          weights=weights, eval_every_iter=eval_every_iter,
                                                          hard_sample_stop=hard_sample_stop, early_stop=early_stop,
                                                          dev_set=(dev_words, dev_tags, dev_pos_tags), model_save=True,
                                                          lr_decay=lr_decay,
                                                          stop_strategy=stop_mode, log=True, return_model=True,
                                                          show_model_status=False)
        else:
            batch_hard_pad, summary, model = tuning_model(model, optimizer, vocab2index,
                                                          (token, token), label_alter, batch_hard_pad,
                                                          summary, tune_epoch=tune_epoch, Type=Type, criterion=criterion,
                                                          batch=use_batch, batch_size=batch_size, protect_epoch=0,
                                                          weights=weights, eval_every_iter=eval_every_iter,
                                                          hard_sample_stop=hard_sample_stop, early_stop=early_stop,
                                                          dev_set=(dev_words, dev_tags, dev_pos_tags), model_save=True,
                                                          lr_decay=lr_decay,
                                                          stop_strategy=stop_mode, log=True, return_model=True,
                                                          show_model_status=False)

        # test the own probability
        token_prob, token_predict = sample_eval(token_lower, vocab2index, model, activation, Type,
                                                show_result=True, return_prob=True)
        token_prob_alter, token_predict_alter = sample_eval(token_lower_alter, vocab2index, model, activation, Type,
                                                show_result=True, return_prob=True)
        model_decision = decisions[token_predict.tolist()[0]]
        decision_probs.append(token_prob[0][0])
        alter_decision_probs.append(token_prob_alter[0][0])
        test_summary = evaluate_wrapper(test_words, test_tags, vocab2index, model, Type, activation, title='Test',
                                        show_dist=True, return_prob=True)
        test2_summary = evaluate_wrapper(test2_words, test2_tags, vocab2index, model, Type, activation, title='Test2',
                                         show_dist=True, return_prob=True)
        
    pdb.set_trace()
