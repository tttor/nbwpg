#!/usr/bin/env python3
import torch, random, sys, os, pickle, argparse, yaml
import numpy as np, pathos.multiprocessing as mp
import gym_util.common_util as cou
import polnet as pn, util_bwopt as u
from collections import defaultdict
from poleval_pytorch import get_rpi_s, get_Ppi_ss, get_ppisteady_s, get_Qsa

def main():
    arg = parse_arg()
    with open(os.path.join(arg.cfg), 'r') as f:
        cfg = yaml.load(f, Loader=u.YAMLCustomLoader)
        cfg = {**cfg['common'], **cfg}; del cfg['common']
        cfg['timestamp'] = u.get_timestamp()
        print(cfg)

    print('making wx, wy mesh...')
    res = float(cfg['resolution'])
    wx = np.arange(cfg['wxmin'], cfg['wxmax'] + res, res)
    wy = np.arange(cfg['wymin'], cfg['wymax'] + res, res)
    wxmat, wymat = np.meshgrid(wx, wy, indexing='ij') # `ij` for 3d mayavi compatibility
    wxmat = np.round(wxmat, cfg['round_decimal'])
    wymat = np.round(wymat, cfg['round_decimal'])
    nrow, ncol = wxmat.shape
    print('wxmat.shape', wxmat.shape)
    # print('sorted(np.unique(wxmat).tolist())', sorted(np.unique(wxmat).tolist()))

    print('making envprop exactly mesh...')
    nrow, ncol = wxmat.shape
    arg_generator = ({**cfg, **{'init_param_x': wxmat[i, j], 'init_param_y': wymat[i, j], 'ij': (i, j)}} \
        for i in range(nrow) for j in range(ncol))

    log_list = []
    if arg.ncpu==1:
        for i, arg_i in enumerate(arg_generator):
            # if not(arg_i['init_param_x']==0. and arg_i['init_param_y']==5.):
            #     continue
            print('i {} ij {} wx {} wy {}'.format(i, arg_i['ij'], arg_i['init_param_x'], arg_i['init_param_y']))
            log = get_envprop(arg_i)
            log_list.append(log)
    else:
        pool = mp.ProcessingPool(ncpus=arg.ncpu)
        log_list = pool.map(get_envprop, arg_generator)

    meshdata_dict = defaultdict(dict)
    for log in log_list:
        for k,v in log.items():
            if k in ['ij', 'env_assetdir']:
                continue
            meshdata_dict[k][log['ij']] = v

    meshdata_mat = {}
    for k, v in meshdata_dict.items():
        v_shape = v[0, 0].shape if (v[0,0].ndim > 0) else (1,)
        meshdata_mat[k] = np.empty(tuple([nrow, ncol] + list(v_shape)))
        for ij, vv in v.items():
            i, j = ij
            meshdata_mat[k][i, j, :] = vv
    meshdata_mat['wxmat'] = wxmat; meshdata_mat['wymat'] = wymat

    envid_short = u.get_shortenvid(cfg['envid'])
    tag = ['meshdata_envprop', 'res{:.2f}'.format(cfg['resolution']), cfg['polnet']['mode']]
    logdir = os.path.join(log_list[0]['env_assetdir'], envid_short, 'data',
        u.make_stamp(tag, cfg['timestamp']))
    os.makedirs(logdir, exist_ok=False)

    for k, v in meshdata_mat.items():
        fname = k + '.pkl'
        meshdata_k = {'cfg': cfg, k: v}
        with open(os.path.join(logdir, fname), 'wb') as f:
            pickle.dump(meshdata_k, f)

    fname = 'cfg.yaml'
    with open(os.path.join(logdir, fname), 'w') as f:
        yaml.dump(cfg, f)

def get_envprop(arg):
    log = {'ij': arg['ij']}
    sfx = arg['polnet']['state_feature_extractor_id']

    env = cou.make_single_env(arg['envid'], arg['seed'])
    nS, nA = env.nS, env.nA; nA_list = env.nA_list
    Psas = torch.tensor(env.get_Psas()).double()
    Rsa = torch.tensor(env.get_Rsa()).double()
    allstatefeature = torch.from_numpy(env.get_allstatefeature(sfx))
    s0 = env.reset()
    log['env_assetdir'] = os.path.join(env.dpath, 'asset')

    PolicyNetwork = pn.policynetclass_dict[arg['polnet']['mode']]
    pi_net = PolicyNetwork(nA_list); pi_net.double()
    n_param = sum([i.numel() for i in pi_net.parameters()])
    init_param = {pi_net.weight_x_name: arg['init_param_x'], pi_net.weight_y_name: arg['init_param_y']}
    for n, p in pi_net.named_parameters():
        p.data.fill_(init_param[n])
        p.data = p.data.double()

    # policy evaluation
    PI = pn.policy_net2tabular(allstatefeature, pi_net, requires_grad=True)
    rpi = get_rpi_s(Rsa, PI); Ppi = get_Ppi_ss(Psas, PI)
    ppi_steady = get_ppisteady_s(Ppi, PI)
    Ppi_steady = torch.vstack([ppi_steady]*nS) # unichain: same rows
    I = torch.eye(nS).double()
    Zpi = torch.inverse(I - Ppi + Ppi_steady) # fundamental matrix
    Hpi = torch.matmul(Zpi, I - Ppi_steady) # deviation matrix
    g = torch.dot(ppi_steady, rpi) # gain
    b = torch.matmul(Hpi, rpi) # bias
    Q = get_Qsa(g, b, Psas, Rsa, nA_list) # action value
    log['bs0'] = b[s0].detach().clone().numpy()
    log['g'] = g.detach().clone().numpy()
    log['PI'] = PI.detach().clone().numpy()

    gamma_list = [np.round(gamma, 2) for gamma in np.arange(0.,1., 0.05)]
    gamma_list += [0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 0.99999999]
    for gamma in gamma_list:
        Dpi = torch.inverse(I - gamma*Ppi) # discounted steady IMPROPER state distrib
        d = torch.matmul(Dpi, rpi) # discounted
        gamma_str = '{:.2f}'.format(gamma) if (gamma <= 0.99) else '{:.8f}'.format(gamma)
        log['_'.join(['ds0', gamma_str])] = d[s0].detach().clone().numpy()

    # grad: auto grad
    grad_g = torch.autograd.grad(g, pi_net.parameters(),
        allow_unused=False, create_graph=True, retain_graph=True)
    grad_g = torch.vstack(grad_g).squeeze()
    log['grad_g'] = grad_g.detach().clone().numpy()

    grad_b = torch.autograd.grad(b[s0], pi_net.parameters(),
        allow_unused=False, create_graph=True, retain_graph=True)
    grad_b = torch.vstack(grad_b).squeeze()
    log['grad_b'] = grad_b.detach().clone().numpy()

    # hess: auto grad
    hess_g = []; hess_b = []
    for pidx in range(n_param):
        mask = torch.zeros(n_param); mask[pidx] = 1.

        hess_g_i = torch.autograd.grad(grad_g, pi_net.parameters(),
            grad_outputs=mask, allow_unused=False, create_graph=False, retain_graph=True)
        hess_g.append(torch.vstack(hess_g_i).squeeze())

        hess_b_i = torch.autograd.grad(grad_b, pi_net.parameters(),
            grad_outputs=mask, allow_unused=False, create_graph=False, retain_graph=True)
        hess_b.append(torch.vstack(hess_b_i).squeeze())
    hess_g = torch.vstack(hess_g); hess_b = torch.vstack(hess_b)
    log['hess_g'] = hess_g.detach().clone().numpy()
    log['hvp_g'] = torch.matmul(torch.pinverse(hess_g), grad_g).detach().clone().numpy()
    log['hess_b'] = hess_b.detach().clone().numpy()
    log['hvp_b'] = torch.matmul(torch.pinverse(hess_b), grad_b).detach().clone().numpy()

    # fisher, tmix
    fisher_atallstate = {s: torch.zeros(n_param, n_param).double() for s in range(nS)}
    for s in range(nS):
        for a in range(nA_list[s]):
            # Do NOT use torch.log(PI[s, a]): unstable grad on extreme values (with sigmoid fn)
            pi = pi_net(torch.from_numpy(env.get_statefeature([s], sfx)))
            prob_a = pi.probs.squeeze(dim=u.sample_dimth)[a]
            logprob_a = pi.log_prob(torch.tensor([a]))
            grad_logpi = torch.autograd.grad(logprob_a, pi_net.parameters(),
                allow_unused=False, create_graph=False, retain_graph=True)
            grad_logpi = torch.vstack(grad_logpi).squeeze(dim=u.feature_dimth)
            fisher_atallstate[s] += prob_a*torch.outer(grad_logpi, grad_logpi)

    fisher_b = torch.zeros((n_param, n_param)).double()
    fisher_mode = arg['fisher_mode']
    if 'fisher_transient_withsteadymul_upto_t' in fisher_mode:
        tabs_hat = 1 + int(fisher_mode.replace('fisher_transient_withsteadymul_upto_t', ''))
    else:
        raise NotImplementedError

    tmix = None; tmix_rtol = env.tmix_cfg['rtol']; tmix_atol = env.tmix_cfg['atol']
    for t in range(env.tmax_xep):
        Ppi_pwr = torch.matrix_power(Ppi, t)

        if torch.allclose(Ppi_steady[s0, :], Ppi_pwr[s0, :], rtol=tmix_rtol, atol=tmix_atol):
            for s in range(nS):
                fisher_b += tabs_hat*Ppi_steady[s0, s]*fisher_atallstate[s]

            tmix = t # specific to s0
            break
        else:
            if t < tabs_hat:
                for s in range(nS):
                    fisher_b += Ppi_pwr[s0, s]*fisher_atallstate[s]
    assert tmix is not None
    log['tmix'] = np.array(tmix)
    log['fisher_b'] = fisher_b.detach().clone().numpy()
    log['fvp_b'] = torch.matmul(torch.pinverse(fisher_b), grad_b).detach().clone().numpy()

    # closure
    return log

def parse_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', help='cfg filepath', type=str, default=None, required=True)
    parser.add_argument('--ncpu', help='number of cpu', type=int, default=None, required=True)
    arg = parser.parse_args()
    arg.cfg = arg.cfg.replace('file://','')
    return arg

if __name__ == '__main__':
    main()
