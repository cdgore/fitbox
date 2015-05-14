import numpy as np
import scipy as sp
import scipy.sparse
# import pandas as pd
import itertools
import mmh3
import datetime
# import math
# import matplotlib.pyplot as plt
# from collections import deque
# import pickle

k = 18


def hash_factory(key):
    k = 18
    trunc_bytes = (1 << k) + (~1 + 1)
    hash_space = (1 << 16) + (~1 + 1)
    hash_salt = [hash_space >> 2]
    hash_seed = reduce(
        lambda x, y: (x * y) % hash_space,
        [int(s.encode('hex'), 16) for s in key] + hash_salt)

    def hash_feature(val):
        return mmh3.hash(val, hash_seed) & trunc_bytes
    return hash_feature


def quad_product(quad, func):
    """ Run a function on the quadratic product of some strings
    inputs:
    quadratic columns in the form [[[a, b], [c, d, e]], [[a, w], [r, x, y, z]]]
    function (a, b) => c to apply
    """
    return reduce(
        lambda x1, x2: x1 + x2,
        map(
            lambda x: [
                reduce(
                    func,
                    y)
                for y in itertools.product(*x)],
            quad))


single_features = [
    'appID',
    'campaignID',
    'connectionType',
    'groupID',
    'locationCountry',
    'locationRegion',
    'momentID']
label = 'target'

quads = [[['appID'], ['campaignID', 'groupID', 'connectionType', 'locationCountry']],
         [['campaignID'], ['locationCountry', 'momentID']]
         ]

quad_features = quad_product(quads, lambda a, b: str(a) + "_" + str(b))
all_features = single_features + quad_features

hash_functions = dict(map(lambda f: (f, hash_factory(f)), all_features))


def get_features(row):
    quad_transformed_items = row.items() + quad_product(
        quads,
        lambda a, b: (str(a) + "_" + str(b), str(row.get(a, '')) + "_" + str(row.get(b, ''))))
    x = map(
        lambda (k, v): hash_functions.get(k)(v),
        quad_transformed_items)
    return np.array([0] + x)


def get_features_sparse(row):
    sparse_features_rows = get_features(row)
    num_features = sparse_features_rows.size
    sparse_features_cols = np.zeros(num_features)
    sparse_features_data = np.ones(num_features)
    return sp.sparse.csc_matrix((sparse_features_data, (sparse_features_rows, sparse_features_cols)),
                                dtype=np.int8,
                                shape=(1 << k, 1))


def update_feature_hash_mapping(row, feat_map):
    quad_transformed_items = row.items() + quad_product(
        quads,
        lambda a, b: (str(a) + "_" + str(b), str(row.get(a, '')) + "_" + str(row.get(b, ''))))
    x = map(
        lambda (k, v): (hash_functions.get(k)(v), str(k) + "_" + str(v)),
        quad_transformed_items)
    return dict([(0, "intercept")] + x)


def parse_row(row):
    y = float(row.get(label, 0.))
    features = dict(filter(lambda x: x[0] != label, row.items()))
    x = get_features_sparse(features)
    results = (y, x)
    return results


def map_feature_hash_to_names(samples, feat_hash):
    for t, row in enumerate(samples):
        y = float(row.get(label, 0.))
        features = dict(filter(lambda x: x[0] != label, row.items()))
        feat_hash.update(update_feature_hash_mapping(features, feat_hash))
        if t % 5000 == 0 and t > 1:
            print('%s\tencountered: %d' % (
                datetime.datetime.now(), t))
    return feat_hash

