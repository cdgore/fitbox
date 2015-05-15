import numpy as np
import scipy as sp
import scipy.sparse as sparse
# import pandas as pd
import itertools
# import math
# import matplotlib.pyplot as plt
from collections import deque
# import pickle

from preprocess.feature_hash import parse_row

k = 18
# Initialize weight vector
w = sparse.csc_matrix((1 << k, 1), dtype=np.float)
alpha = 0.1

# Regularization parameter
l2_reg = 0.4
intercept = 0.0
tau = 10000.


def train(samples, w):
    """Online lbfgs training"""
    batch_size = 10000
    l2_r = 0.04
    lamb_const = 1.
#     c = 0.1
    c = 1.0
    m = 10 # size of (s, y) ring buffer
    mini_batch_buffer = deque([])
    for t, row in enumerate(samples):
        mini_batch_buffer.append(parse_row(row))
        if (t + 1) % batch_size == 0 and t > 1:
            print 'batch: ' + str((t + 1) / batch_size)
            if t + 1 == batch_size:
                # For the first batch, set the intercept to line up with the CTR
                w = set_first_intercept(w, list(mini_batch_buffer))
                print w
            w_g_l = olbfgs_batch(list(mini_batch_buffer), w, l2_r, m, c, lamb_const)
            return (w_g_l[0], w_g_l[1], mini_batch_buffer, w_g_l[2], w_g_l[3])
            mini_batch_buffer = deque([])
#         y_hat = lr_predict(w, x)
#         w = update_gradient(y, y_hat, w, x)
        
#         loss += logloss(y_hat, y)
#         if t % 5000 == 0 and t > 1:
#             print('%s\tencountered: %d\tcurrent logloss: %f' % (
#                 datetime.datetime.now(), t, loss/t))
#     return w


def set_first_intercept(w, X1):
    ctr_counts = reduce(
        lambda a, b: (a[0] + b[0], a[1] + b[1]),
        map(
            lambda x: (x[0], 1),
            X1))
    ctr_ratio = float(ctr_counts[0]) / float(ctr_counts[1]) + 10**-6
    new_intercept = -np.log(ctr_ratio**-1 - 1)
    w[0, 0] = new_intercept
    return w


def set_first_intercept_spark(w, X1):
    ctr_counts = X1.map(
        lambda x: (x[0], 1)
    ).reduce(
        lambda a, b: (a[0] + b[0], a[1] + b[1])
    )
    ctr_ratio = float(ctr_counts[0]) / float(ctr_counts[1]) + 10**-6
    new_intercept = -np.log(ctr_ratio**-1 - 1)
    w[0, 0] = new_intercept
    return w
