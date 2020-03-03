import numpy as np
import matplotlib.pyplot as plt

def show_gridsearch(result_dict, savefig=False):
    '''
    :param result_dict: each key is a label and corresponding value
           is the result sequence
    :return: no return
    '''
    for key, value_seq in result_dict.items():
        plt.plot(value_seq, label=key)
        max_value_index = np.argmax(value_seq)
        max_value = value_seq[max_value_index]
        plt.annotate(s=str(max_value_index) + ', ' + str(max_value),
                     xy=(max_value_index, max_value))

    plt.xlabel('threshold')
    plt.ylabel('/'.join(list(result_dict.keys())))
    plt.title('Grid Search result of ', '/'.join(list(result_dict.keys())))
    plt.legend()
    if savefig:
        plt.savefig('./report/IncrementalTest/gridSearchResult.png')
    plt.show()


