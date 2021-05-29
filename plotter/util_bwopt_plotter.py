import numpy as np

# These follow the standard way to organize data in a matrix of `n_sample x n_feature`
sample_dimth = 0
feature_dimth = action_dimth = 1

def arrange_mesh2mat(xs, ys, zs, ndecimal):
    x_sorted = sorted(np.unique(np.round(xs, ndecimal)).tolist())
    y_sorted = sorted(np.unique(np.round(ys, ndecimal)).tolist())
    zmat = np.empty(xs.shape)
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            col_idx = x_sorted.index(np.round(xs[i, j], ndecimal))
            row_idx = y_sorted.index(np.round(ys[i, j], ndecimal))
            zmat[row_idx, col_idx] = zs[i, j]
    return (zmat, x_sorted, y_sorted)

def print_stat(zs, tag=None):
    if tag is not None:
        print('===== print_stat: ', tag, '=====')
    # assert np.isfinite(zs).all()
    # print('mean {}'.format(np.nanmean(zs)))
    # print('std {}'.format(np.nanstd(zs)))
    # print('min {} max {}'.format(np.amin(zs), np.amax(zs)))
    # print('argmin {} argmax {}'.format(np.argmin(zs), np.argmax(zs)))
    print('mean {}'.format(np.mean(zs)))
    print('std {}'.format(np.std(zs)))
    print('max-min range {} min {} max {} '.format(np.max(zs) - np.min(zs), np.min(zs), np.max(zs)))
    for p in range(0, 100 + 1, 5):
        # print('percentile', p, ':', np.nanpercentile(zs.flatten(), p))
        print('percentile', p, ':', np.percentile(zs.flatten(), p))

def get_shortenvid(envid):
    return envid.split(':')[1]

def get_angular_err(pred, groundtruth):
    groundtruth_norm = np.linalg.norm(groundtruth, ord=None)
    groundtruth_u = groundtruth.data
    if groundtruth_norm > 0.:
        groundtruth_u = groundtruth/groundtruth_norm

    pred_norm = np.linalg.norm(pred, ord=None)
    pred_u = pred.data
    if pred_norm > 0.:
        pred_u = pred/pred_norm

    dot = np.dot(groundtruth_u, pred_u).item()
    dot = min(dot, 1.); dot = max(-1., dot) # keep in [-1, 1] range
    ang = np.arccos(dot)
    return ang
