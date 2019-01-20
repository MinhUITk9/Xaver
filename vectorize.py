import json
import sys
import os

import logging as log
import multiprocessing as mp
import constant

import numpy as np
from tqdm import tqdm
import hashlib

from features import FeatureExtractor


def raw_feature_iterator(file_paths):
    for path in file_paths:
        with open(path, "r") as f:
            for line in f:
                yield line


def vectorize_data(arg):
    row, raw_data, x_path, y_path, n_rows = arg
    extractor = FeatureExtractor()
    dim = FeatureExtractor.dim
    raw_features = json.loads(raw_data)
    y = np.memmap(y_path, dtype=np.float32, mode="r+", shape=n_rows)
    y[row] = raw_features["label"]
    feature_vector = extractor.process_raw_features(raw_features)
    x = np.memmap(x_path, dtype=np.float32, mode="r+", shape=(n_rows, dim))
    x[row] = feature_vector


def sha256_checksum(file_name, block_size=65536):
    if not os.path.isfile(file_name):
        return None
    sha256 = hashlib.sha256()
    with open(file_name, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()


def VectorData(subset):
    # Checking dataset
    data_dir = os.path.join(os.getcwd(), 'ember')
    if subset == 'train':
        paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(6)]
        n_rows = constant.TRAIN_NUM_FILE
    elif subset == 'test':
        paths = [os.path.join(data_dir, "test_features.jsonl"), ]
        n_rows = constant.TEST_NUM_FILE

    for p in paths:
        if not os.path.exists(p):
            log.info('File not found: {}'.format(p))
            sys.exit(1)
    x_path = os.path.join(data_dir, "X_{}.dat".format(subset))
    y_path = os.path.join(data_dir, "y_{}.dat".format(subset))

    if os.path.exists(x_path + '.shd256') and os.path.exists(y_path + '.shd256'):
        with open(x_path + '.shd256', 'r') as f:
            x_checksum = f.read()
        with open(y_path + '.shd256', 'r') as f:
            y_checksum = f.read()
        if x_checksum == sha256_checksum(x_path) and y_checksum == sha256_checksum(y_path):
            log.info('"{}" subset is vectorized'.format(subset))
            return

    dim = FeatureExtractor.dim
    x = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(n_rows, dim))
    y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=n_rows)
    del x, y

    log.info('vectoring samples in "{}" subset'.format(subset))
    pool = mp.Pool()
    arg_iterator = ((row, raw_data, x_path, y_path, n_rows)
                    for row, raw_data in enumerate(raw_feature_iterator(paths)))
    for _ in tqdm(pool.imap_unordered(vectorize_data, arg_iterator),
                  unit='row', unit_scale=True, ncols=96, miniters=1, total=n_rows):
        pass

    x_checksum = sha256_checksum(x_path)
    with open(x_path + '.shd256', 'w') as f:
        f.write(x_checksum)
    y_checksum = sha256_checksum(y_path)
    with open(y_path + '.shd256', 'w') as f:
        f.write(y_checksum)


def SaveToFile(subset):
    data_dir = os.path.join(os.getcwd(), 'ember')
    if subset == 'train':
        n_rows = constant.TRAIN_NUM_FILE
    elif subset == 'test':
        n_rows = constant.TEST_NUM_FILE
    else:
        log.error('subset must be "train" or "test"')
        sys.exit(1)

    x_npy = os.path.join(data_dir, "X_{}.npy".format(subset))
    y_npy = os.path.join(data_dir, "y_{}.npy".format(subset))

    if os.path.exists(x_npy + '.shd256') and os.path.exists(y_npy + '.shd256'):
        with open(x_npy + '.shd256', 'r') as f:
            x_checksum = f.read()
        with open(y_npy + '.shd256', 'r') as f:
            y_checksum = f.read()
        if x_checksum == sha256_checksum(x_npy) and y_checksum == sha256_checksum(y_npy):
            log.info('Numpy files of "{}" subset is existed!'.format(subset))
            return

        log.info('Saving labeled samples to numpy files in "{}" subset'.format(subset))
    dim = FeatureExtractor.dim
    x_dat = os.path.join(data_dir, "X_{}.dat".format(subset))
    y_dat = os.path.join(data_dir, "y_{}.dat".format(subset))
    x = np.memmap(x_dat, dtype=np.float32, mode="r", shape=(n_rows, dim))
    y = np.memmap(y_dat, dtype=np.float32, mode="r", shape=n_rows)
    labeled_rows = (y != -1)
    np.save(x_npy, x[labeled_rows])
    np.save(y_npy, y[labeled_rows])

    x_checksum = sha256_checksum(x_npy)
    with open(x_npy + '.shd256', 'w') as f:
        f.write(x_checksum)
    y_checksum = sha256_checksum(y_npy)
    with open(y_npy + '.shd256', 'w') as f:
        f.write(y_checksum)


def VectorDataset():
    VectorData('train')
    SaveToFile('train')
    VectorData('test')
    SaveToFile('test')
    log.info('Dataset is vectorized')
