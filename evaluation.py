from sklearn.metrics import precision_recall_fscore_support


def metrics_cal(predicts, tags, detail=False, sklearn_mode='macro'):
    # number of negative prediction
    negative_pred = len(predicts) - sum(predicts)
    acc = sum([x == y for x, y in zip(predicts, tags)])

    if not sklearn_mode:
        # tag = 1, pred = 0
        FN = sum([1 for pred, tag in zip(predicts, tags) if pred - tag == -1])
        # tag = 0, pred = 1
        FP = sum([1 for pred, tag in zip(predicts, tags) if pred - tag == 1])
        TP = sum(predicts) - FP
        TN = negative_pred - FN

        if detail:
            print("TP: {} FP: {} TN: {} FN: {}".format(TP, FP, TN, FN))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2 * precision * recall) / (precision + recall)

    else:
        precision, recall, f1, _ = precision_recall_fscore_support(tags, predicts, average=sklearn_mode)

    accuracy = acc / len(tags)

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    iphone_label = [1 for _ in range(70)] + [0 for _ in range(30)]
    iphone_pred = [1 for _ in range(40)] + [0 for _ in range(30)] + \
                  [1 for _ in range(20)] + [0 for _ in range(10)]

    # Expect 0.667, 0.5714, 0.6153
    print(precision_recall_fscore_support(iphone_label, iphone_pred, average='binary'))

    # Expect 0.4583, 0.4523, 0.4505
    print(precision_recall_fscore_support(iphone_label, iphone_pred, average='macro'))

    # TP  = 40, FP = 20, TN=10, FN=30
    acc, precision, recall, F1 = metrics_cal(iphone_pred, iphone_label, detail=True)
    print('Metric Testing' + ": acc: {} precision: {} recall: {} F1: {}".format(
        round(acc, 2), round(precision, 2),
        round(recall, 2), round(F1, 2)))


