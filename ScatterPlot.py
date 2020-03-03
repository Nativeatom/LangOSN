import os
import matplotlib.pyplot as plt
import numpy as np

def TestResultScatter(test_words, test_tags, metrics, threshold, train_volume, save_path='', note='', show=True):
    fig = plt.figure(figsize=(24,10))
    annotate_shift = 0.01
    colors = {'correct-correct': 'lime', 'correct-wrong':'blue', 'wrong-correct': 'orange', 'wrong-wrong': 'purple'}
    pred_label2tag = {0: {1: 'correct-wrong', 0:'correct-correct'}, 1: {0: 'wrong-correct', 1:'wrong-wrong'}}
    test_logprobs = metrics[-1]

    ptsgroup = {'correct-wrong':[], 'correct-correct':[], 'wrong-correct':[], 'wrong-wrong':[]}

    # threshold line
    plt.axhline(y=threshold, ls='--', c="red")
    plt.annotate(s='logprob={}'.format(round(threshold, 2)), xy=(5, threshold+annotate_shift))

    for index, (word, label, prob) in enumerate(zip(test_words, test_tags, test_logprobs)):
        prediction = int(prob <= threshold)
        # group the point belongs to
        g = pred_label2tag[prediction][label]
        plt.scatter(index, prob, color=colors[g])
        plt.annotate(s='{} {}'.format(word, round(prob, 2)), xy=(index, prob+annotate_shift))

    title = 'Probability Distribution - Language Model - Test Set - train{} - Acc {} - F1 {}'.format(train_volume,
                                                                                                      round(metrics[0], 2),
                                                                                                      round(metrics[3], 2))
    plt.title(title)
    plt.ylabel('Loglikelihood')
    if len(save_path):
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        plt.savefig('{}/{}-{}.png'.format(save_path, title, note))
    if show:
        plt.show()