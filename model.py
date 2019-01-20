import datetime
import logging as log
import os
import time

import lightgbm as lgb
import matplotlib
import numpy
from sklearn import metrics

from histogram import SaveROCImage, SaveDatasetImage

matplotlib.use("Agg")



def TrainDataset():
    modelPath = os.path.join(os.getcwd(), "model.txt")
    dataDir = os.path.join(os.getcwd(), "ember")

    numpy.random.seed(2018)
    params = {
        'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
        'objective': 'binary',
        'num_trees': 400,
        'num_leaves': 64,
        'learning_rate': 0.05,
        'num_threads': 24,
        'min_data': 2000,
    }

    x_train = numpy.load(os.path.join(dataDir, "X_train.npy"), mmap_mode='r')
    y_train = numpy.load(os.path.join(dataDir, "y_train.npy"), mmap_mode='r')
    log.info('Number of training samples: {}'.format(y_train.shape[0]))

    log.info('Start training the model!')
    start_time = time.time()
    dataset = lgb.Dataset(x_train, y_train)
    model = lgb.train(params, dataset)
    training_time = time.time() - start_time
    log.info('Training time: {}'.format(datetime.timedelta(seconds=training_time)))
    model.save_model(modelPath)
    log.info('Model saved at: {}'.format(modelPath))


def EvaluateTestSample():
    model_path = os.path.join(os.getcwd(), "model.txt")
    roc_curve_path = os.path.join(os.getcwd(), 'roc_curve.png')
    dataset_path = os.path.join(os.getcwd(), 'dataset.png')
    data_dir = os.path.join(os.getcwd(), "ember")
    numpy.random.seed(2018)

    X_test = numpy.load(os.path.join(data_dir, "X_test.npy"), mmap_mode='r')
    y_test = numpy.load(os.path.join(data_dir, "y_test.npy"), mmap_mode='r')
    log.info('Number of testing samples: {}'.format(y_test.shape[0]))

    model = lgb.Booster(model_file=model_path)

    y_pred = model.predict(X_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

    roc = metrics.auc(fpr, tpr)
    idx_1 = (numpy.abs(0.001 - fpr[numpy.where((0.001 - fpr) >= 0)])).argmin()
    idx_2 = (numpy.abs(0.01 - fpr[numpy.where((0.01 - fpr) >= 0)])).argmin()
    acc_1 = metrics.accuracy_score(y_test, numpy.where(y_pred >= threshold[idx_1], 1, 0))
    acc_2 = metrics.accuracy_score(y_test, numpy.where(y_pred >= threshold[idx_2], 1, 0))

    TN, FP, FN, TP = metrics.confusion_matrix(y_test, numpy.where(y_pred >= 0.5, 1, 0)).ravel()
    fpr_0 = FP / (FP + TN)
    tpr_0 = TP / (TP + FN)
    acc_0 = (TP + TN) / (TP + FP + FN + TN)

    log.info("Area Under ROC Curve     : {:.6f}".format(roc))
    log.info("=====   Threshold at 0.5    =====")
    log.info("False Alarm Rate         : {:2.4f} %".format(fpr_0 * 100))
    log.info("Detection Rate           : {:2.4f} %".format(tpr_0 * 100))
    log.info("Overall Accuracy         : {:2.4f} %".format(acc_0 * 100))
    log.info("=====  FPR less than 0.1%   =====")
    log.info("Detection Rate           : {:2.4f} %".format(tpr[idx_1] * 100))
    log.info("Overall Accuracy         : {:2.4f} %".format(acc_1 * 100))
    log.info("Threshold                : {:.6f}".format(threshold[idx_1]))
    log.info("=====  FPR less than 1.0%   =====")
    log.info("Detection Rate           : {:2.4f} %".format(tpr[idx_2] * 100))
    log.info("Overall Accuracy         : {:2.4f} %".format(acc_2 * 100))
    log.info("Threshold                : {:.6f}".format(threshold[idx_2]))

    # Save Result to image
    SaveROCImage(fpr, tpr, roc, idx_1, idx_2, roc_curve_path)
    SaveDatasetImage(dataset_path)
