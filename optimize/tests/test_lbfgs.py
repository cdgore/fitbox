import pytest
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import math
from scipy.optimize import rosen, rosen_der

from optimize.olbfgs import olbfgs_batch, rosen_obj_func, rosen_obj_func_grad, sgd_with_momentum


def test_rosen():
    m_rosen = 20
    c_rosen = 1.
    lamb_const_rosen = 1.
    w_rosen = sparse.csc_matrix([1.3, 0.7, 0.8, 1.9, 1.2]).T

    w_g_l_rosen = olbfgs_batch(w_rosen, rosen_obj_func,
                               rosen_obj_func_grad, m_rosen, c_rosen,
                               lamb_const_rosen)
    w_hat = w_g_l_rosen[0].toarray()[0]
    assert np.sum(np.round(w_hat) - np.ones(np.shape(w_hat))) == 0


def test_sgd():
    w_rosen = sparse.csc_matrix([1.3, 0.7, 0.8, 1.9, 1.2]).T

    w_2_rosen, previous_delta = sgd_with_momentum(w_rosen,
                                                  rosen_obj_func_grad,
                                                  0.0002,
                                                  0.7,
                                                  max_num_iter=1000,
                                                  norm_2_grad_threshold=1e-4)
    w_hat = w_2_rosen.toarray()[0]
    assert np.sum(np.round(w_hat) - np.ones(np.shape(w_hat))) == 0
