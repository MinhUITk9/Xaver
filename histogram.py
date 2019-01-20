import matplotlib.pyplot as pyplot
import numpy


def SaveROCImage(fpr, tpr, roc, idx_1, idx_2, path):
    pyplot.title('ROC Curve')
    pyplot.ylim([0.8, 1])
    pyplot.xscale('log')
    pyplot.plot(fpr, tpr, 'b', label='AUC = {:0.6f}'.format(roc))
    tpr_1 = tpr[idx_1]
    pyplot.plot([10 ** -3, 10 ** -3], [0, tpr_1], 'r')
    pyplot.plot([0, 10 ** -3], [tpr_1, tpr_1], 'r')
    tpr_2 = tpr[idx_2]
    pyplot.plot([10 ** -2, 10 ** -2], [0, tpr_2], 'r')
    pyplot.plot([0, 10 ** -2], [tpr_2, tpr_2], 'r')
    pyplot.legend(loc='lower right')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.savefig(path)


def SaveDatasetImage(path):
    pyplot.clf()
    maliciousNum = (100000, 300000)
    benignNum = (100000, 300000)
    ind = numpy.arange(2)
    width = 0.5
    pyplot.figure(figsize=(6, 8))
    p1 = pyplot.bar(ind, maliciousNum, width)
    p2 = pyplot.bar(ind, benignNum, width, bottom=maliciousNum)
    pyplot.xlabel('Subset')
    pyplot.xticks(ind, ('test', 'train'))
    pyplot.yticks(numpy.arange(1, 7) * 100000, ['{}K'.format(i) for i in numpy.arange(1, 7) * 100])
    pyplot.legend((p1[0], p2[0]), ('malicious', 'benign'), prop={'size': 16}, loc='upper left')
    pyplot.savefig(path)

