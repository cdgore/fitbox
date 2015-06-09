import pytest
import numpy as np
import scipy as sp
import scipy.sparse as sparse

from preprocess.FeatureManager import HashFeatureManager, NumericFeatureManager


def test_hash_feature_manager():
    hash_feats = HashFeatureManager() \
        .set_k(22) \
        .set_label('target') \
        .set_single_features([
            'fooID',
            'barID',
            'connectionType',
            'bazID',
            'locationCountry',
            'locationRegion',
            'favoriteColor']) \
        .set_quad_features([
            [['fooID'], ['barID', 'bazID', 'connectionType', 'locationCountry']],
            [['barID'], ['locationCountry', 'favoriteColor']]
        ])

    row1 = {
        'fooID': 'f1d2d2f924e986ac86fdf7b36c94bcdf32beec15',
        'barID': 'e242ed3bffccdf271b7fbaf34ed72d089537b42f',
        'connectionType': 'WIFI',
        'bazID': '6eadeac2dade6347e87c0d24fd455feffa7069f0',
        'locationCountry': 'US',
        'locationRegion': 'MI',
        'favoriteColor': 'blue',
        'target': 1.0
        }

    parsed_row = hash_feats.parse_row(row1)

    assert parsed_row[0] == 1.
    assert parsed_row[1].nnz == 14


def test_numeric_feature_manager():
    numeric_feats = NumericFeatureManager() \
        .set_label('click') \
        .set_features([
            'foo',
            'bar',
            'baz'])

    binary_row1 = {
        'click': 1.0,
        'foo': -0.526,
        'bar': 0.115,
        'baz': -0.123
        }

    parsed_row = numeric_feats.parse_row(binary_row1)

    w_1 = sparse.csc_matrix((np.ones(4), (np.array(range(0, 4)), np.zeros(4))),
                            dtype=np.float,
                            shape=(4, 1))
    x_1 = parsed_row[1]
    wTx_1 = w_1.T.dot(x_1)[0, 0]
    expected_inner_prod = 1. + -0.526 + 0.115 + -0.123

    assert parsed_row[0] == 1.
    assert wTx_1 == expected_inner_prod
