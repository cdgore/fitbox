import numpy as np
# from numpy import sqrt
import scipy as sp
import scipy.sparse as sparse
import math
from collections import deque
from scipy.optimize import rosen, rosen_der
import datetime


def line_search(a_0=1., b=0.01, tau=0.5):
    """Backtracking-Armijo"""
    def backtrack(f, g, w, p):
        g_w = g(w)
        a = a_0
        # l = 0
        f_w = f(w)
        f_w_ap = f(w + a * p)
        # print "f(w) + a * b * g_w.T.dot(p)[0, 0] < f(w + a * p): " + str(f(w) + a * b * g_w.T.dot(p)[0, 0])
        # print "f(w + a * p) : " + str(f(w + a * p))
        while f_w + a * b * g_w.T.dot(p)[0, 0] < f_w_ap:
            a = tau * a
            # l += 1
            # f_w = f(w)
            f_w_ap = f(w + a * p)
            print a
        return (a, f_w)
    return backtrack


def olbfgs_batch(w, obj_func, obj_func_grad, m, c, lamb_const,
                 batch_size=10000, gradient_estimates=None, B_t=None,
                 grad_norm2_threshold=10**-5, max_iter=1000, min_iter=0.,
                 line_search_init=20.):
    print "Run online lbfgs over one batch of data"
    # batch_size = len(X1)
    eta_0 = float(batch_size) / (float(batch_size) + 2.)

    def get_tau(tau_p=None):
        if tau_p is None:
            tau_p = 0.
        return float(1 * 10**2) * (2.**int(tau_p))

    tau_pow = 0
    tau_counter = 0
    tau = get_tau(tau_pow)
    if gradient_estimates is None:
        gradient_estimates = deque([])
    # while not converging
    grad_norm2 = float('inf')
    losses = []
    t = 0
    starttime = datetime.datetime.now()
    grad = obj_func_grad(w)
    endtime = datetime.datetime.now()
    print 'Time for compute grad = %s ' % (endtime - starttime)
    pre_grad = grad

    while (grad_norm2 > grad_norm2_threshold and t < max_iter) or t < min_iter:
        starttime = datetime.datetime.now()
        print 'Iteration %s, starttime= %s' % (t + 1, starttime)
        grad = pre_grad
        p_t = -grad

        # Update direction with low-rank estimate of Hessian
        if len(gradient_estimates) > 0:
            p_t = lbfgs_direction_update(list(gradient_estimates), grad, t)

        cur_loss = float('inf')
        last_loss = 0.
        stochastic_line_search_count = 0
        eta_t = 1.
        if t == 0:
            eta_t, cur_loss = line_search(a_0=line_search_init, b=0.1, tau=0.75)(obj_func, obj_func_grad, w, p_t)
        else:
            last_loss = losses[-1]
            while (cur_loss > last_loss):
                if (stochastic_line_search_count > 0):
                    tau_pow -= 1
                    tau_counter = 0
                    print 'Failed to improve objective function, cutting Tau in half'

                tau = get_tau(tau_pow)
                eta_t = (tau / (tau + float(t + 1))) * eta_0
                s_t = (eta_t / c) * p_t  # change in parameters
                w_tp1 = w.copy() + s_t
                cur_loss = obj_func(w_tp1)
                print 'cur_loss: ' + str(cur_loss)
                print 'last_loss: ' + str(last_loss)
                stochastic_line_search_count += 1

        tau = get_tau(tau_pow)
        if t > 0:
            eta_t = (tau / (tau + float(t + 1))) * eta_0
        s_t = (eta_t / c) * p_t  # change in parameters
        w_tp1 = w + s_t

        # Keep track of how long it's been since an adjustment to tau
        if tau_counter >= m and tau_pow < 0:
            tau_pow += 1
            tau_counter = 0
            print 'Increasing tau power'
        tau_counter += 1

        grad_tp1 = obj_func_grad(w_tp1)
        pre_grad = grad_tp1

        y_t = grad_tp1 - grad + lamb_const * s_t  # change in gradients
        gradient_estimates.append((s_t, y_t))
        while len(gradient_estimates) > m:
            gradient_estimates.popleft()
        w = w_tp1
#         B_t = update_B_t((s_t, y_t), B_t=B_t, c=(c / eta_t))
        B_t = update_B_t((s_t, y_t), B_t=B_t, c=10**-4)
        t += 1
#         cur_loss = calc_loss(X1, w)
        losses.append(cur_loss)
        # print 'Losses: ' + str(losses)
        grad_norm2 = grad_tp1.T.dot(grad_tp1)[0, 0]
        print 'Norm-2(gradient(loss_func)): ' + str(grad_norm2)
        endtime = datetime.datetime.now()
        # print 'Iteration [%s],endtime= %s' % (t,endtime)
        print 'Time for Iteration %s = %s ' %(t,( endtime - starttime ) )

    return (w, gradient_estimates, losses, B_t, grad_norm2)


def lbfgs_direction_update(s_y_list, grad, t):
    p_t = -grad
    epsilon = 10**-10
    s_list = [r[0] for r in s_y_list]
    y_list = [r[1] for r in s_y_list]
    rho = map(
        lambda x: (x[0].T.dot(x[1])[0, 0]) ** -1,
        s_y_list)
    alpha = map(
        lambda rs: rs[0] * rs[1].T.dot(p_t)[0, 0],
        zip(rho, s_list))
    p_t2 = p_t - reduce(
        lambda x, y: x + y,
        map(
            lambda r: r[0] * r[1],
            zip(alpha, y_list)))
    p_t3 = p_t2
    if(t == 0):
        p_t3 = epsilon * p_t2
    else:
        p_t3 = p_t2 * (1. / float(len(s_y_list))) * reduce(
            lambda x, y: x + y,
            map(
                lambda s_y: s_y[0].T.dot(s_y[1])[0, 0] /
                s_y[1].T.dot(s_y[1])[0, 0],
                s_y_list)
        )
    beta = map(
        lambda r_y: r_y[0] * r_y[1].T.dot(p_t3)[0, 0],
        zip(rho, y_list))
    a_minus_b = map(lambda x: float(x[0]) - float(x[1]), zip(alpha, beta))
    return p_t3 + reduce(
        lambda x, y: x + y,
        map(
            lambda r: r[0] * r[1],
            zip(a_minus_b, s_list)))


def sgd_with_momentum(w, obj_func_grad, eta, rho, max_num_iter=10,
                      norm_2_grad_threshold=1e-4, previous_param_delta=None):
    """
    Stochastically update the parameters of the objective function
    Runs until threshold of norm_2(gradient) or maximum number of iterations
    reached

    w: weight vector
    obj_func_grad: gradient of objective function
    eta: learning rate
    rho: momentum constant (set to 0. to turn off momentum updates)
    max_num_iter: maximum number of iterations
    norm_2_grad_threshold: minimum threshold for norm2 of the gradient
    previous_param_delta: previous change in weights

    returns (new parameters, delta of parameters)

    """
    if previous_param_delta is None:
        previous_param_delta = sparse.csc_matrix(w.shape, dtype=np.float)
    norm_2_grad = float('inf')
    num_iter = 0

    while (num_iter < max_num_iter and norm_2_grad > norm_2_grad_threshold):
        print "Iteration number: %d" % (num_iter + 1)
        grad = obj_func_grad(w)
        param_delta = rho * previous_param_delta - eta * grad
        w = w + param_delta
        previous_param_delta = param_delta
        norm_2_grad = grad.T.dot(grad)[0, 0]
        print "Norm2(gradient): %f" % (norm_2_grad)
        num_iter += 1
    return (w, previous_param_delta)


def lr_row_loss(w, x, y):
    f = w.T.dot(x)[0, 0]
    f_clipped = max(-20., min(20., f))
    y_scaled = 2. * y - 1.  # Target scaled to {-1, 1} for logistic loss
    return -np.log(logistic_function(y_scaled * f_clipped))


def lr_row_gradient(w, x, y):
    f = w.T.dot(x)[0, 0]
    f_clipped = max(-20., min(20., f))
    y_scaled = 2. * y - 1.  # Target scaled to {-1, 1} for logistic loss
    return y_scaled * x * (logistic_function(y_scaled * f_clipped) - 1.)


def lr_row_hess_diag(w, x, y):
    f = w.T.dot(x)[0, 0]
    f_clipped = max(-20., min(20., f))
    y_scaled = 2. * y - 1.  # Target scaled to {-1, 1} for logistic loss
    phi_y_w_T_x = logistic_function(y_scaled * f_clipped)
    D_i = phi_y_w_T_x * (1. - phi_y_w_T_x)
    return (x * D_i).multiply(x)


def sparse_min_max(sparse_matrix, lb, ub):
    sparse_matrix = sparse_matrix.tocsc()
    indices = sparse_matrix.indices
    indptr = sparse_matrix.indptr
    data = np.array([max(float(lb), min(float(ub), f))
                     for f
                     in sparse_matrix.data])
    return sp.sparse.csc_matrix(
        (data, indices, indptr),
        shape=sparse_matrix.shape,
        dtype=np.float)


def sparse_logistic_function(sparse_vector):
    dense_array = sparse_vector.T.toarray()[0]
    len_dense_array = len(dense_array)
    indices = np.array(xrange(0, len_dense_array))
    indptr = np.array([0, len_dense_array])
    data = np.array([logistic_function(f)
                     for f
                     in dense_array])

    return sp.sparse.csc_matrix(
        (data, indices, indptr),
        shape=(len_dense_array, 1),
        dtype=np.float)


def make_i(sparse_matrix):
    sparse_matrix = sparse_matrix.tocsc()
    indices = sparse_matrix.indices
    indptr = sparse_matrix.indptr
    data = np.ones(indices.size)
    return sp.sparse.csc_matrix(
        (data, indices, indptr),
        shape=sparse_matrix.shape,
        dtype=np.float)


def s_lr_row_loss(w, x, y):
    f = w.T.dot(x)
    f_clipped = f_clipped = sparse_min_max(f, -20., 20.)
    y_scaled = (2. * y) + (-1. * make_i(y))  # Target scaled to {-1, 1} for logistic loss
    _logistic = sparse_logistic_function(y_scaled.multiply(f_clipped))
    _logistic.data = -np.log(_logistic.data)
    return _logistic


def s_lr_row_gradient(w, x, y):
    f = w.T.dot(x)
    f_clipped = sparse_min_max(f, -20., 20.)
    y_scaled = (2. * y) + (-1. * make_i(y))  # Target scaled to {-1, 1} for logistic loss
    phi = sparse_logistic_function(y_scaled.multiply(f_clipped))
    return x.dot(((phi + (-1. * make_i(phi))).multiply(y_scaled)).T)


# def s_lr_row_hess_diag(w, x, y):
#     f = w.T.dot(x)[0, 0]
#     f_clipped = max(-20., min(20., f))
#     y_scaled = 2. * y - 1.  # Target scaled to {-1, 1} for logistic loss
#     phi_y_w_T_x = logistic_function(y_scaled * f_clipped)
#     D_i = phi_y_w_T_x * (1. - phi_y_w_T_x)
#     return (x * D_i).multiply(x)


def logistic_function(t):
    if t > 0:
        return 1. / (1. + np.exp(-t))
    else:
        return np.exp(t) / (1. + np.exp(t))


def lr_predict(w, x):
    wTx = w.transpose().dot(x)[0, 0]
    # wTx_clipped = max(20., min(-20., wTx))
    return logistic_function(wTx)


def make_mr_obj_func(X, row_obj_func, reg_modifier=None):
    if reg_modifier is None:
        reg_modifier = lambda a, b: a

    def mr_obj_func(w):
        obj_func_result = reduce(
            lambda a, b: a + b,
            map(
                lambda row: row_obj_func(w, row[1], row[0]),
                X)) / float(len(X))
        return reg_modifier(obj_func_result, w)
    return mr_obj_func


def make_mr_gradient(X1, row_gradient, reg_modifier=None):
    if reg_modifier is None:
        reg_modifier = lambda a, b: a

    def partial_mr_gradient(w):
        grad = reduce(
            lambda x, y: x + y,
            map(
                lambda row: row_gradient(w, row[1], row[0]),
                X1
            )
        ) / float(len(X1))
        return reg_modifier(grad, w)
    return partial_mr_gradient


def make_spark_mr_function(X, row_function, zero_val_func, reg_modifier=None):
    if reg_modifier is None:
        reg_modifier = lambda a, b: a
    X_len = X.count()

    def mr_function(w):
        zero_value = zero_val_func(w)
        # X.checkpoint()
        function_result = X.map(
            lambda row: row_function(w, row[1], row[0]),
            preservesPartitioning=True
        ).fold(
            zero_value,
            lambda a, b: a + b
        ) / float(X_len)
        return reg_modifier(function_result, w)
    return mr_function


def make_spark_mr_obj_func(X, row_obj_func, reg_modifier=None):
    zero_val_func = lambda w: 0.
    return make_spark_mr_function(X, row_obj_func, zero_val_func, reg_modifier)


def make_spark_mr_gradient(X, row_gradient, reg_modifier=None):
    zero_val_func = lambda w: sparse.csc_matrix(w.shape, dtype=np.float)
    return make_spark_mr_function(X, row_gradient, zero_val_func, reg_modifier)


def make_l2_reg(l2_r=None, intercept_index=0):
    if l2_r is None:
        l2_r = 0.01

    def partial_reg(loss_func, w):
        w_reg = w.copy()  # only add regularization penalty on non-intercept weights
        if intercept_index is not None:
            w_reg[intercept_index, 0] = 0.0
        return loss_func + 0.5 * l2_r * (w_reg.T.dot(w_reg)[0, 0])
    return partial_reg


def make_l2_reg_gradient(l2_r=None, intercept_index=0):
    if l2_r is None:
        l2_r = 0.01

    def partial_reg(grad, w):
        w_reg = w.copy()  # only add regularization penalty on non-intercept weights
        if intercept_index is not None:
            w_reg[intercept_index, 0] = 0.0
        return grad + l2_r * w_reg
    return partial_reg


def make_l2_reg_hessian_diag(l2_r=None, intercept_index=0):
    if l2_r is None:
        l2_r = 0.01

    def partial_reg(hess, w):
        w_reg = w.copy()  # only add regularization penalty on non-intercept weights
        if intercept_index is not None:
            w_reg[intercept_index, 0] = 0.0
        w_reg_data = w_reg.nonzero()[0]
        w_reg_nnz = w_reg_data.shape[0]
        l2_r_I = sparse.csc_matrix(
            (
                np.ones(w_reg_nnz) * l2_r,
                (w_reg_data, np.zeros(w_reg_nnz))
            ),
            shape=(w_reg.shape[0], 1))
        return hess + l2_r_I
    return partial_reg


def make_lr_l2_obj_func(X, l2_r=None):
    l2_reg = make_l2_reg(l2_r)
    return make_mr_obj_func(X, lr_row_loss, l2_reg)


def make_s_lr_l2_obj_func(X, l2_r=None):
    l2_reg = make_l2_reg(l2_r)
    return make_mr_obj_func(X, s_lr_row_loss, l2_reg)


def make_lr_l2_gradient(X, l2_r=None):
    l2_reg = make_l2_reg_gradient(l2_r)
    return make_mr_gradient(X, lr_row_gradient, l2_reg)


def make_s_lr_l2_gradient(X, l2_r=None):
    l2_reg = make_l2_reg_gradient(l2_r)
    return make_mr_gradient(X, s_lr_row_gradient, l2_reg)


def make_spark_lr_l2_obj_func(X, l2_r=None):
    l2_reg = make_l2_reg(l2_r)
    return make_spark_mr_obj_func(X, lr_row_loss, l2_reg)


def make_spark_lr_l2_gradient(X, l2_r=None):
    l2_reg = make_l2_reg_gradient(l2_r)
    return make_spark_mr_gradient(X, lr_row_gradient, l2_reg)


def make_spark_s_lr_l2_obj_func(X, l2_r=None):
    l2_reg = make_l2_reg(l2_r)
    return make_spark_mr_obj_func(X, s_lr_row_loss, l2_reg)


def make_spark_s_lr_l2_gradient(X, l2_r=None):
    l2_reg = make_l2_reg_gradient(l2_r)
    return make_spark_mr_gradient(X, s_lr_row_gradient, l2_reg)


def make_spark_lr_l2_hessian_diag(X, l2_r=None):
    l2_reg = make_l2_reg_hessian_diag(l2_r)
    return make_spark_mr_gradient(X, lr_row_hess_diag, l2_reg)


# ========================= Batch updates ============================
# bu: batch update
# Add regularization to cost function
# output : func which returns a scalar number
def make_l2_reg_bu(Q, M, intercept_index=0):
    def partial_reg(loss_func, w):
        w_reg = w.copy() - M  # only add regularization penalty on non-intercept weights
        if intercept_index is not None:
            w_reg[intercept_index, 0] = 0.0
        return loss_func + 0.5 * (Q.T.dot(w_reg.multiply(w_reg))[0, 0])
    return partial_reg


def make_l2_reg_gradient_bu(Q, M, intercept_index=0):
    '''
    l2 regularized gradient batch update
    input : csc_matrix, csc_matrix
    output : func which returns csc_matrix
    '''
    def partial_reg(grad, w):
        w_reg = w.copy() - M  # only add regularization penalty on non-intercept weights
        if intercept_index is not None:
            w_reg[intercept_index, 0] = 0.0
        return grad + Q.multiply(w_reg)
    return partial_reg


# def make_spark_lr_l2_hessian_diag_bu(X, l2_r=None):
#     l2_reg = make_l2_reg_hessian_diag(l2_r)
#     return make_spark_mr_gradient(X, lr_row_hess_diag, l2_reg)


def make_spark_lr_l2_obj_func_bu(Q, M, X):
    l2_reg = make_l2_reg_bu(Q, M)
    return make_spark_mr_obj_func(X, lr_row_loss, l2_reg)


def make_spark_lr_l2_gradient_bu(Q, M, X):
    l2_reg = make_l2_reg_gradient_bu(Q, M)
    return make_spark_mr_gradient(X, lr_row_gradient, l2_reg)


def make_spark_s_lr_l2_obj_func_bu(Q, M, X):
    l2_reg = make_l2_reg_bu(Q, M)
    return make_spark_mr_obj_func(X, s_lr_row_loss, l2_reg)


def make_spark_s_lr_l2_gradient_bu(Q, M, X):
    l2_reg = make_l2_reg_gradient_bu(Q, M)
    return make_spark_mr_gradient(X, s_lr_row_gradient, l2_reg)



# def calc_gradient_rosen(X1, w, l2_r):
#     return sparse.csc_matrix(rosen_der(w.T.toarray()[0])).T


def rosen_obj_func(w):
    return rosen(w.T.toarray()[0])


def rosen_obj_func_grad(w):
    return sparse.csc_matrix(rosen_der(w.T.toarray()[0])).T


def row_loss(w, x, y):
    f = w.T.dot(x)[0, 0]
    f_clipped = max(-20., min(20., f))
    y_scaled = 2. * y - 1.  # Target scaled to {-1, 1} for logistic loss
    return -np.log(logistic_function(y_scaled * f_clipped))


def calc_loss(X1, w):
    return reduce(
        lambda a, b: a + b,
        map(
            lambda row: row_loss(w, row[1], row[0]),
            X1)) / float(len(X1))


def update_B_t(tup, B_t=None, c=None):
    if c is None:
        c = 0.0
#     epsilon = 10**-4
    epsilon = 1.0
    s_t = tup[0]
    y_t = tup[1]
    s_length = s_t.shape[0]
    if B_t is None:
        s_t_data = s_t.nonzero()[0]
        s_t_nnz = s_t.getnnz()
        B_t = sparse.csc_matrix((np.ones(s_t_nnz) * epsilon, (s_t_data, np.zeros(s_t_nnz))), shape=(s_length, 1))
    rho = (s_t.T.dot(y_t)[0, 0])**-1
    left_hand_side = rho * (s_t.multiply(y_t))
    right_hand_side = rho * (y_t.multiply(s_t))
    B_tp1 = B_t - B_t.multiply(left_hand_side) \
        - B_t.multiply(right_hand_side) \
        + B_t.multiply(right_hand_side).multiply(left_hand_side) \
        + c * rho * s_t.multiply(s_t)
    return B_tp1
