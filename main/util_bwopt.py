import time, socket, datetime, os, json, yaml
import numpy as np, git
from typing import Any, IO

# These follow the standard way to organize data in a matrix of `n_sample x n_feature`
sample_dimth = 0
feature_dimth = action_dimth = 1

### optim util #################################################################
def backtracking_linesearch_ascent(p, fval, grad, stepdir, fn, niter):
    # an inexact line search method
    # Nocedal, p37, Algorithm 3.1 (Backtracking Line Search).
    # Boyd, p464, Algorithm 9.2 Backtracking line search
        # kappa \in (0, 0.5), rho \in (0, 1), alpha0 = 1
    alpha = 1.0; kappa = 1e-4; rho = 0.5; assert np.isfinite(fval)
    dirder = np.dot(grad, stepdir)
    # assert dirder >= 0.,'Direction is descent!' # 'dirder > 0` does NOT handle PSD Fisher
    if dirder < 0:
        return 0. # Handle stochastic stepdir, which may be a descent
    for _ in range(niter):
        fval_next = fn(p + alpha*stepdir, compute_derivative=False)
        fval_next = fval_next.detach().numpy().item()
        if not(np.isfinite(fval_next)) or fval_next < (fval + kappa*alpha*dirder):
            # infinite `fval_next` may come from the log barrier
            alpha = rho*alpha # shrink using the contraction factor: rho
        else:
            return alpha
    return 0. # raise RuntimeError('backtracking_linesearch: running out niter!')

### logging util ###############################################################
def load_cfg(cfg_fpath, arg):
    if (len(arg.custom)%2) == 0:
        custom = {arg.custom[i]:arg.custom[i+1] for i in range(0, len(arg.custom), 2)}
    else:
        raise RuntimeError('len(arg.custom)', len(arg.custom))
    with open(os.path.join(cfg_fpath), 'r') as f:
        cfg = yaml.load(f, Loader=YAMLCustomLoader)
        cfg = {**cfg['common'], **cfg,**vars(arg), **custom}; del cfg['common']
        cfg['logdir'] = os.path.join(os.path.expanduser("~"), cfg['logdir'])
        cfg['timestamp'] = get_timestamp()
        cfg['gitlog'] = get_gitinfo(cfg['repo'],
            allow_diff=('logdir' in cfg.keys()) and ('tmp' in cfg['logdir']))
        return cfg

def get_wxwymesh(cfg):
    # 2D mesh of weight x (wx) and weight y (wy)
    res = float(cfg['resolution'])
    wx = np.arange(cfg['wxmin'], cfg['wxmax'] + res, res)
    wy = np.arange(cfg['wymin'], cfg['wymax'] + res, res)
    wxmat, wymat = np.meshgrid(wx, wy, indexing='ij') # `ij` for 3d mayavi compatibility
    wxmat = np.round(wxmat, cfg['round_decimal'])
    wymat = np.round(wymat, cfg['round_decimal'])
    return (wxmat, wymat)

def get_hostname():
    hostname = socket.gethostname(); hostname = hostname.split('.')[0]
    return hostname

def make_stamp(tag, timestamp=None):
    if timestamp is None: timestamp = get_timestamp(fracsec=False)
    str_list = list(filter(None, tag + [get_hostname(), timestamp]))
    stamp = '__'.join(str_list)
    return stamp

def get_timestamp(time=None, fracsec=True):
    if time is None: time = datetime.datetime.now()
    if fracsec:
        stamp = time.strftime("%Y%m%d-%H%M%S-%f")
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
    return stamp

def get_shortenvid(envid):
    return envid.split(':')[1]

def get_gitinfo(repodir_list, allow_diff):
    info = {};  max_msg_char = 80
    for p in repodir_list:
        repo = git.Repo(path=p); name = os.path.basename(p)
        if len(repo.head.commit.diff(None)) > 0:
            if allow_diff:
                print(p, ': ALLOWS difftree against workingtree!')
            else:
                raise RuntimeError(p+': difftree against workingtree EXISTS!!!')
        info[name] = {'path': p, 'sha': '{}'.format(repo.head.object.hexsha),
            'msg': repo.head.object.message.strip()[0:max_msg_char],
            'time': '{}'.format(time.asctime(time.localtime(repo.head.object.committed_date)))}
    return info

def get_angular_err(pred, groundtruth):
    # /home/tor/ws/bwopt-plotter/util_bwopt_plotter.py
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

################################################################################
# https://stackoverflow.com/questions/528281/how-can-i-include-a-yaml-file-inside-another
# https://gist.github.com/joshbode/569627ced3076931b02f
class YAMLCustomLoader(yaml.SafeLoader):
    def __init__(self, stream: IO) -> None:
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir
        super().__init__(stream)

def yaml_construct_include(loader: YAMLCustomLoader, node: yaml.Node) -> Any:
    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')
    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, YAMLCustomLoader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())

yaml.add_constructor('!include', yaml_construct_include, YAMLCustomLoader)
################################################################################
