import numpy as np
import scipy as sp
import scipy.sparse as sparse

from abc import ABCMeta, abstractmethod

import itertools
import mmh3
import datetime


class FeatureManager(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def parse_row(self, row):
        return sparse.csc_matrix([])

    @abstractmethod
    def get_w(self):
        return sparse.csc_matrix([])

    def flatten_list(self, rec, level=None, tail_list=None):
        if level is None:
            level = 0
        if tail_list is None:
            tail_list = []
        tail_list.append(str(level))
        for r in rec:
            if type(r) == list:
                tail_list = self.flatten_list(r, level + 1, tail_list)
            else:
                tail_list.append(str(r))
        return tail_list


class HashFeatureManager(FeatureManager):
    def __init__(self):
        self.k = 0
        self.single_features = []
        self.quad_features = []
        self.quads = []
        self.label = ''
        self.all_features = []
        self.hash_functions = {}

    def __key(self):
        return reduce(
            lambda x, y: x + y,
            self.flatten_list([
                self.k,
                self.single_features,
                self.quad_features,
                self.quads,
                self.label,
                self.all_features

            ])
        )

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return mmh3.hash(self.__key(), 5)

    def set_k(self, new_k):
        self.k = new_k
        return self

    def set_label(self, new_label):
        self.label = new_label
        return self

    def set_single_features(self, new_sf):
        self.single_features = new_sf
        self.all_features = self.single_features + self.quad_features
        self.hash_functions = dict(
            map(
                lambda f: (f, self.hash_factory(f)),
                self.all_features))
        return self

    def set_quad_features(self, new_quads):
        self.quads = new_quads
        self.quad_features = self.quad_product(
            self.quads,
            lambda a, b: str(a) + "_" + str(b))
        self.all_features = self.single_features + self.quad_features
        self.hash_functions = dict(
            map(
                lambda f: (f, self.hash_factory(f)),
                self.all_features))
        return self

    def hash_factory(self, key):
        trunc_bytes = (1 << self.k) + (~1 + 1)
        hash_space = (1 << 16) + (~1 + 1)
        hash_salt = [hash_space >> 2]
        hash_seed = reduce(
            lambda x, y: (x * y) % hash_space,
            [int(s.encode('hex'), 16) for s in key] + hash_salt)

        def hash_feature(val):
            return mmh3.hash(val, hash_seed) & trunc_bytes
        return hash_feature

    def quad_product(self, quad, func):
        """ Run a function on the quadratic product of some strings
        inputs:
        quadratic columns in the form
            [
                [[a, b], [c, d, e]],
                [[a, w], [r, x, y, z]]
            ]
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

    def get_features(self, row):
        quad_transformed_items = row.items() + self.quad_product(
            self.quads,
            lambda a, b: (str(a) + "_" + str(b), str(row.get(a, '')) + "_" + str(row.get(b, ''))))
        x = map(
            lambda (k, v): self.hash_functions.get(k, lambda a: None)(v),
            quad_transformed_items)
        return np.array([0] + filter(lambda y: y is not None, x))

    def get_features_sparse(self, row):
        sparse_features_rows = self.get_features(row)
        num_features = sparse_features_rows.size
        sparse_features_cols = np.zeros(num_features)
        sparse_features_data = np.ones(num_features)
        return sparse.csc_matrix((sparse_features_data, (sparse_features_rows, sparse_features_cols)),
                                 dtype=np.int8,
                                 shape=(1 << self.k, 1))

    def update_feature_hash_mapping(self, row, feat_map={}):
        quad_transformed_items = row.items() + self.quad_product(
            self.quads,
            lambda a, b: (str(a) + "_" + str(b), str(row.get(a, '')) + "_" + str(row.get(b, ''))))
        x = map(
            lambda (k, v): (self.hash_functions.get(k, lambda a: None)(v), str(k) + "_" + str(v)),
            quad_transformed_items)
        return dict([(0, "intercept")] + filter(lambda y: y is not None, x))

    def parse_row(self, row):
        y = float(row.get(self.label, 0.))
        features = dict(filter(lambda x: x[0] != self.label, row.items()))
        x = self.get_features_sparse(features)
        results = (y, x)
        return results

    def get_w(self):
        return sp.sparse.csc_matrix(
            (1 << self.k, 1),
            dtype=np.float)

    def map_feature_hash_to_names(self, samples, feat_hash={}):
        for t, row in enumerate(samples):
            # y = float(row.get(label, 0.))
            features = dict(filter(lambda x: x[0] != self.label, row.items()))
            feat_hash.update(self.update_feature_hash_mapping(features, feat_hash))
            if t % 5000 == 0 and t > 1:
                print('%s\tencountered: %d' % (
                    datetime.datetime.now(), t))
        return feat_hash

    def map_feature_hash_to_names_spark(self, samples, feat_hash={}):
        def seq_op(feat_hash, row):
            features = dict(filter(lambda x: x[0] != self.label, row.items()))
            feat_hash.update(self.update_feature_hash_mapping(features, feat_hash))
            return feat_hash

        def comb_op(feat_hash1, feat_hash2):
            for i in feat_hash2.items():
                feat_hash1.update({i})
            return feat_hash1

        return samples.aggregate(
            feat_hash,
            seq_op,
            comb_op
        )


class NumericFeatureManager(FeatureManager):
    def __init__(self):
        self.k = 0
        self.label = ''
        self.all_features = []

    def set_k(self, new_k):
        self.k = new_k
        return self

    def set_label(self, new_label):
        self.label = new_label
        return self

    def set_features(self, new_features):
        self.all_features = new_features
        if self.k == 0:
            self.k = len(self.all_features)
        return self

    def parse_row(self, row):
        y = float(row.get(self.label, 0.))
        simple_features = [1.] + map(
            lambda f: row.get(f, None),
            self.all_features)
        x_data_all = filter(
            lambda y: y[1] is not None,
            zip(
                range(0, len(simple_features)),
                simple_features))
        x_data = np.array([x[1] for x in x_data_all])
        x_rows = np.array([x[0] for x in x_data_all])
        x_cols = np.zeros(len(x_data_all))

        x = sparse.csc_matrix((x_data, (x_rows, x_cols)),
                              dtype=np.float,
                              shape=(self.k + 1, 1))
        results = (y, x)
        return results

    def get_w(self):
        return sp.sparse.csc_matrix(
            (self.k + 1, 1),
            dtype=np.float)
