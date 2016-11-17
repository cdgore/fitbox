import numpy as np


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
    tmp_w = w.copy()
    ctr_counts = X1.map(
        lambda x: (x[0], 1),
        preservesPartitioning=True
    ).fold(
        (0, 0),
        lambda a, b: (a[0] + b[0], a[1] + b[1])
    )
    numer = ctr_counts[0]
    denom = ctr_counts[1]
    if denom == 0:
        denom = 1
    ctr_ratio = numer / denom
    # new_intercept = -ln(ctr_ratio**-1 - 1)
    ctr_ratio.data = -np.log([
        ((_ + 10**-6) ** -1) - 1.
        for _ in ctr_ratio.data])
    new_intercept = ctr_ratio
    tmp_w[0] = new_intercept.T
    return tmp_w
