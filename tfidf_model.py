from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pdb
from config import *
import numpy as np
import string
from OnlineSpellingServer import build_vocab
vectorizer = TfidfVectorizer()

def get_doc(sents):
    doc_info = []
    for i, sent in enumerate(sents):
        count = len(sent)
        temp = {'doc_id': i, 'doc_length': count}
        doc_info.append(temp)
    return doc_info


def computeTF(doc_info, freqDict_list):
    # tf = (frequency of the term in the doc/ total number of terms in the doc)
    TF_scores = []
    for tempDict in freqDict_list:
        id_ = tempDict['doc_id']
        for k in tempDict['freq_dict']:
            temp = {'doc_id': id_, 'TF_score': tempDict['freq_dict'][k] / doc_info[id_ - 1]['doc_length'],
                    'key': k}  # use id-1 for doc_info?
            TF_scores.append(temp)
    return TF_scores


def computeIDF(doc_info, freqDict_list, smooth_idf=True, smooth_log=True):
    # idf = ln(total number of docs/number of docs with term in it)
    IDF_scores = []
    for counter, _dict_ in enumerate(freqDict_list):
        for k in _dict_['freq_dict'].keys():
            count = sum([k in tempDict['freq_dict'] for tempDict in freqDict_list]) + smooth_idf
            temp = {'doc_id': counter, 'IDF_score': np.log((len(doc_info) + smooth_idf) / count) + smooth_log, 'key': k}
            IDF_scores.append(temp)
    return IDF_scores


def computeTFIDF(TF_scores, IDF_scores):
    # n_sample is number of documents
    # n_feature is number of vocabularies
    TFIDF_scores = []
    for j in IDF_scores:
        for i in TF_scores:
            if j['key'] == i['key'] and j['doc_id'] == i['doc_id']:
                temp = {'doc_id': j['doc_id'],
                        'TFIDF_score': j['IDF_score'] * i['TF_score'],
                        'key': i['key']}
        TFIDF_scores.append(temp)
    return TFIDF_scores


def computeTFIDF_matrix(tfidf_scores, vocab2index, n_sample, n_feature, normalize=True):
    tfidf_matrix = np.zeros((n_sample, n_feature))
    for tfidf_score in tfidf_scores:
        tfidf_matrix[tfidf_score['doc_id'], vocab2index[tfidf_score['key']]] = tfidf_score['TFIDF_score']
    if normalize:
        tfidf_matrix = tfidf_matrix / np.linalg.norm(tfidf_matrix, ord=2, axis=1, keepdims=True)
    return tfidf_matrix


def tfidf_feature(docs, voca2indx, max_token_length, smooth_idf=True, smooth_log=True, combine_pos=True):
    # docs is a list of docs, each with a list of tokens
    doc_info = get_doc(docs)
    freqDict_list = create_freq_dict(docs)
    tf_scores = computeTF(doc_info, freqDict_list)
    idf_scores = computeIDF(doc_info, freqDict_list, smooth_idf, smooth_log)
    tfidf_scores = computeTFIDF(tf_scores, idf_scores)
    tfidf_matrix = computeTFIDF_matrix(tfidf_scores, voca2indx, n_sample=len(docs), n_feature=len(voca2indx),
                                       normalize=True)

    if combine_pos:
        pos_matrix = np.zeros((len(docs), max_token_length))
        for doc_index, token in enumerate(docs):
            token_list = token
            for token_index, token_element in enumerate(token_list):
                pos_matrix[doc_index, token_index] = tfidf_matrix[doc_index, voca2indx[token_element]]
        combined_matrix = np.concatenate((tfidf_matrix, pos_matrix), axis=1)
        return combined_matrix

    else:
        return tfidf_matrix


def create_freq_dict(sents):
    # create frequency dictionary for each word in each document
    freqDict_list = []
    #     vocab2index = {}
    for i, sent in enumerate(sents):
        freq_dict = {}
        for word in sent:
            #             if word not in vocab2index.keys():
            #                 vocab2index[word] = len(vocab2index)
            freq_dict[word] = freq_dict.get(word, 0) + 1
        temp = {'doc_id': i, 'freq_dict': freq_dict}
        freqDict_list.append(temp)
    #     index2vocab = {value:key for key, value in vocab2index.items()}
    return freqDict_list





if __name__ == "__main__":
    current_lang = 'en'
    lang = 'English'
    max_token_length = 16

    langCode = lang2code.get(lang, '')
    current_lang = langCode
    vocab2index, index2vocab = build_vocab(langCode)

    # byrule set
    with open("./files/English/city/New_York_City_byrule_label_200_position_test_set.txt", "r") as fp:
        byrule_set = fp.readlines()
        byrule_set = [x.strip('\n').strip(string.punctuation).split() for x in byrule_set]
    fp.close()

    byrule_words = [x[0].lower() for x in byrule_set]
    byrule_tags = [int(x[1]) for x in byrule_set]

    train_set = byrule_words

    train_tags = byrule_tags
    train_character = [' '.join(x[0]).split() for x in train_set]

    # doc_info = get_doc([x.split() for x in train_character])
    # freqDict_list = create_freq_dict([x.split() for x in train_character])
    # tf_scores = computeTF(doc_info, freqDict_list)
    # idf_scores = computeIDF(doc_info, freqDict_list, smooth_idf=True, smooth_log=True)
    # tfidf_scores = computeTFIDF(tf_scores, idf_scores)
    # tfidf_matrix = computeTFIDF_matrix(tfidf_scores, vocab2index, n_sample=len(train_character), n_feature=len(vocab2index),
    #                                    normalize=True)
    #
    # pos_matrix = np.zeros((len(train_character), max_token_length))
    # for doc_index, token in enumerate(train_character):
    #     token_list = token.split()
    #     for token_index, token_element in enumerate(token_list):
    #         pos_matrix[doc_index, token_index] = tfidf_matrix[doc_index, vocab2index[token_element]]

    combined_matrix = tfidf_feature(train_character, vocab2index, max_token_length,
                                    smooth_idf=True, smooth_log=True, combine_pos=True)

    pdb.set_trace()
