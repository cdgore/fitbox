import pytest
import numpy as np
import scipy as sp
import scipy.sparse
import math
from scipy.optimize import rosen, rosen_der

from optimize.olbfgs import olbfgs_batch, rosen_obj_func, rosen_obj_func_grad


def test_rosen():
    m_rosen = 20
    c_rosen = 1.
    lamb_const_rosen = 1.
    w_rosen = sp.sparse.csc_matrix([1.3, 0.7, 0.8, 1.9, 1.2]).T

    w_g_l_rosen = olbfgs_batch(np.zeros(10000), w_rosen, rosen_obj_func,
                               rosen_obj_func_grad, m_rosen, c_rosen,
                               lamb_const_rosen)
    w_hat = w_g_l_rosen[0].toarray()[0]
    assert np.sum(np.round(w_hat) - np.ones(np.shape(w_hat))) == 0
