import numpy as np
import scipy as sp
import scipy.sparse
import math
from collections import deque
from scipy.optimize import rosen, rosen_der


def olbfgs_batch(X1, w, l2_r, m, c, lamb_const,
                 gradient_estimates=None, B_t=None):
    """Run online lbfgs over one batch of data"""
    batch_size = len(X1)
    eta_0 = float(batch_size) / (float(batch_size) + 2.)

    def get_tau(tau_p=None):
        if tau_p is None:
            tau_p = 0.
        return float(1 * 10**2) * (2.**int(tau_pow))

    tau_pow = 0
    tau = get_tau(tau_pow)
    if gradient_estimates is None:
        gradient_estimates = deque([])
    # while not converging
    grad_norm2 = float('inf')
    losses = []
    t = 0
    while (grad_norm2 > 10**-5 and t < 1000):
        grad = calc_gradient(X1, w, l2_r)
        print 'Iteration ' + str(t)
        p_t = -grad

        # Update direction with low-rank estimate of Hessian
        if len(gradient_estimates) > 0:
            p_t = lbfgs_direction_update(list(gradient_estimates), grad, t)

        cur_loss = float('inf')
        last_loss = 0.
        if (t > 0):
            last_loss = losses[-1]
        stochastic_line_search_count = 0
        while (cur_loss > last_loss):
            if (stochastic_line_search_count > 0):
                tau_pow -= 1
                print 'Failed to improve objective function, cutting Tau in half'
            tau = get_tau(tau_pow)
            eta_t = (tau / (tau + float(t))) * eta_0
            s_t = (eta_t / c) * p_t # change in parameters
            w_tp1 = w + s_t
            cur_loss = calc_loss(X1, w_tp1)
            if (t == 0):
                last_loss = cur_loss
            stochastic_line_search_count += 1
        
        tau = get_tau(tau_pow)
        eta_t = (tau / (tau + float(t))) * eta_0
        s_t = (eta_t / c) * p_t # change in parameters
        w_tp1 = w + s_t
            
        grad_tp1 = calc_gradient(X1, w_tp1, l2_r)
        y_t = grad_tp1 - grad + lamb_const * s_t # change in gradients
        gradient_estimates.append((s_t, y_t))
        while len(gradient_estimates) > m:
            gradient_estimates.popleft()
        w = w_tp1
        B_t = update_B_t((s_t, y_t), B_t=B_t, c=(c / eta_t))
        t += 1
#         cur_loss = calc_loss(X1, w)
        losses.append(cur_loss)
        print 'Losses: ' + str(losses)
        grad_norm2 = grad.T.dot(grad)[0, 0]
        print 'Norm-2(gradient(loss_func)): ' + str(grad_norm2)
    return (w, gradient_estimates, losses, B_t)


def olbfgs_batch_rosen(X1, w, l2_r, m, c, lamb_const,
                       gradient_estimates=None, B_t=None):
    """Run online lbfgs over one batch of data"""
    batch_size = len(X1)
    eta_0 = float(batch_size) / (float(batch_size) + 2.)
    
    def get_tau(tau_p=None):
        if tau_p is None:
            tau_p = 0.
        return float(1 * 10**2) * (2.**int(tau_pow))
    
    tau_pow = 0
    tau_counter = 0
    tau = get_tau(tau_pow)
    if gradient_estimates is None:
        gradient_estimates = deque([])
    # while not converging
    grad_norm2 = float('inf')
    losses = []
    t = 0
    while (grad_norm2 > 10**-5 and t < 800):
        grad = calc_gradient_rosen(X1, w, l2_r)
#         print 'Iteration ' + str(t)
        p_t = -grad

        # Update direction with low-rank estimate of Hessian
        if len(gradient_estimates) > 0:
            p_t = lbfgs_direction_update(list(gradient_estimates), grad, t)

        cur_loss = float('inf')
        last_loss = 0.
        if (t > 0):
            last_loss = losses[-1]
        stochastic_line_search_count = 0
        t2 = 0
        # last_loss = loss(w, x)
        while (cur_loss > last_loss and t2 < 30):
            if (stochastic_line_search_count > 0):
                tau_pow -= 1
                tau_counter = 0
                print 'Failed to improve objective function, cutting Tau in half'
#             else:
#                 tau_pow = 0
            tau = get_tau(tau_pow)
            eta_t = (tau / (tau + float(t))) * eta_0
            s_t = (eta_t / c) * p_t # change in parameters
#             print s_t
            w_tp1 = w.copy() + s_t
#             print w.T.toarray()[0]
#             print w_tp1.T.toarray()[0]
            cur_loss = rosen(w_tp1.T.toarray()[0])
#             print 'cur_loss: ' + str(cur_loss)
#             print 'last_loss: ' + str(last_loss)
            if (t == 0):
                last_loss = cur_loss
            stochastic_line_search_count += 1
            t2 += 1
        
        tau = get_tau(tau_pow)
        eta_t = (tau / (tau + float(t))) * eta_0
        if(t == 0):
            eta_t = 10**-8
        s_t = (eta_t / c) * p_t # change in parameters
        w_tp1 = w + s_t
        
        # Keep track of how long it's been since an adjustment to tau
        if tau_counter >= m and tau_pow < 0:
            tau_pow += 1
            tau_counter = 0
            print 'Increasing tau power'
        tau_counter += 1
            
        grad_tp1 = calc_gradient_rosen(X1, w_tp1, l2_r)
        y_t = grad_tp1 - grad + lamb_const * s_t # change in gradients
        gradient_estimates.append((s_t, y_t))
        while len(gradient_estimates) > m:
            gradient_estimates.popleft()
        w = w_tp1
#         B_t = update_B_t((s_t, y_t), B_t=B_t, c=(c / eta_t))
        B_t = update_B_t((s_t, y_t), B_t=B_t, c=10**-4)
        t += 1
#         cur_loss = calc_loss(X1, w)
        losses.append(cur_loss)
#         print 'Losses: ' + str(losses)
        grad_norm2 = grad.T.dot(grad)[0, 0]
#         print 'Norm-2(gradient(loss_func)): ' + str(grad_norm2)
    return (w, gradient_estimates, losses, B_t)


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


def row_gradient(w, x, y):
    f = w.T.dot(x)[0, 0]
    f_clipped = max(-20., min(20., f))
    y_scaled = 2. * y - 1.  # Target scaled to {-1, 1} for logistic loss
    return y_scaled * x * (logistic_function(y_scaled * f_clipped) - 1.)


def logistic_function(t):
    if t > 0:
        return 1. / (1. + np.exp(-t))
    else:
        return np.exp(t) / (1. + np.exp(t))


def lr_predict(w, x):
    wTx = w.transpose().dot(x)[0, 0]
    # wTx_clipped = max(20., min(-20., wTx))
    return 1. / (1. + math.exp(-wTx))


def calc_gradient(X1, w, l2_r):
    w_reg = w.copy()  # only add regularization penalty on non-intercept weights
    w_reg[0, 0] = 0.0
    result1 = reduce(
        lambda x, y: x + y,
        map(
            lambda row: row_gradient(w, row[1], row[0]),
            X1
        )
    ) / float(len(X1))
#     print result1
#     print w_reg
    result2 = result1 + l2_r * w_reg
#     print result2
    return result2


def calc_gradient_rosen(X1, w, l2_r):
    return sp.sparse.csc_matrix(rosen_der(w.T.toarray()[0])).T


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