#!/usr/bin/env python3
import argparse, os, sys, pickle
import numpy as np, pathos.multiprocessing as mp, torch
import gym_util.common_util as cou, polnet as pn, util_bwopt as u
from collections import defaultdict
from poleval_pytorch import get_rpi_s, get_Ppi_ss, get_ppisteady_s, get_Qsa

def main():
    arg = parse_arg()
    cfg = u.load_cfg(arg.cfg, arg)
    wxmat, wymat = u.get_wxwymesh(cfg)
    print(cfg); print('wxmat.shape', wxmat.shape)
    # print('sorted(np.unique(wxmat).tolist())', sorted(np.unique(wxmat).tolist()))

    print('making arg_generator...')
    nrow, ncol = wxmat.shape
    arg_generator = ({**cfg, **{'init_param_x': wxmat[i, j], \
        'init_param_y': wymat[i, j], 'ij': (i, j)}} \
        for i in range(nrow) for j in range(ncol))

    print('making grad_b samplingbased expression exactly mesh...')
    log_list = []
    if arg.ncpu==1:
        for i, cfg_i in enumerate(arg_generator):
            # if not(cfg_i['init_param_x']==0. and cfg_i['init_param_y']==5.):
            #     continue
            print('i {}/{} ij {} wx {} wy {}'.format(i+1, nrow*ncol, cfg_i['ij'],
                cfg_i['init_param_x'], cfg_i['init_param_y']))
            finalinfo = get_gradbias_samplingbased_exactly(cfg_i)
            log_list.append(finalinfo)
    else:
        pool = mp.ProcessingPool(ncpus=cfg['ncpu'])
        log_list = pool.map(get_gradbias_samplingbased_exactly, arg_generator)

    print('meshdata...')
    meshdata = defaultdict(dict)
    for log in log_list:
        for k,v in log.items():
            if k=='ij':
                continue
            meshdata[k][log['ij']] = v
    meshdata['wxmat'] = wxmat; meshdata['wymat'] = wymat; meshdata['cfg'] = cfg
    meshdata = dict(meshdata)

    print('writing...')
    envid_short = u.get_shortenvid(cfg['envid'])
    datadir = os.path.join(log_list[0]['assetdir'], envid_short, 'data')
    os.makedirs(datadir, exist_ok=True)
    tag = ['meshdata_gradbias_samplingbased_exactly',
        'res{:.2f}'.format(cfg['resolution']), cfg['polnet']['mode']]
    fname = u.make_stamp(tag, cfg['timestamp']) + '.pkl'
    with open(os.path.join(datadir, fname), 'wb') as f:
        pickle.dump(meshdata, f)

def get_gradbias_samplingbased_exactly(arg):
    log = defaultdict(list); log['ij'] = arg['ij']
    sfx = arg['polnet']['state_feature_extractor_id']

    env = cou.make_single_env(arg['envid'], arg['seed'])
    nS, nA = env.nS, env.nA; nA_list = env.nA_list
    Psas = torch.tensor(env.get_Psas()).double()
    Rsa = torch.tensor(env.get_Rsa()).double()
    s0 = env.reset() # assume to be deterministic
    allstatefeature = torch.from_numpy(env.get_allstatefeature(sfx))
    log['assetdir'] = os.path.join(env.dpath, 'asset')

    PolicyNetwork = pn.policynetclass_dict[arg['polnet']['mode']]
    pi_net = PolicyNetwork(nA_list); pi_net.double()
    n_param = sum([i.numel() for i in pi_net.parameters()])
    init_param = {pi_net.weight_x_name: arg['init_param_x'],
        pi_net.weight_y_name: arg['init_param_y']}
    for n, p in pi_net.named_parameters():
        p.data.fill_(init_param[n])
        p.data = p.data.double()

    # policy evaluation
    PI = pn.policy_net2tabular(allstatefeature, pi_net, requires_grad=True)
    rpi = get_rpi_s(Rsa, PI); Ppi = get_Ppi_ss(Psas, PI)
    ppi_steady = get_ppisteady_s(Ppi, PI)
    Ppi_steady = torch.vstack([ppi_steady]*nS) # unichain: same rows
    Zpi = torch.inverse(torch.eye(nS) - Ppi + Ppi_steady) # fundamental matrix
    Hpi = torch.matmul(Zpi, torch.eye(nS) - Ppi_steady) # deviation matrix
    g = torch.dot(ppi_steady, rpi) # gain
    b = torch.matmul(Hpi, rpi) # bias
    Q = get_Qsa(g, b, Psas, Rsa, nA_list) # action value

    # Exact via autograd
    grad_b = torch.autograd.grad(b[s0], pi_net.parameters(),
        allow_unused=False, create_graph=True, retain_graph=True)
    grad_b = torch.vstack(grad_b).squeeze()
    grad_b_np = grad_b.detach().numpy()

    grad_g = torch.autograd.grad(g, pi_net.parameters(),
        allow_unused=False, create_graph=False, retain_graph=True)
    grad_g = torch.vstack(grad_g).squeeze()

    # premix part
    tmix = None; tmix_rtol = env.tmix_cfg['rtol']; tmix_atol = env.tmix_cfg['atol']
    premix = torch.zeros(n_param).double() # accumulator for premix terms
    for t in range(env.tmax_xep):
        Ppi_pwr = torch.matrix_power(Ppi, t)
        premix_t = torch.zeros(n_param).double()
        for s in range(nS):
            for a in range(nA_list[s]):
                pi = pi_net(torch.from_numpy(env.get_statefeature([s], sfx)))
                logprob_a = pi.log_prob(torch.tensor([a]))
                grad_logpi = torch.autograd.grad(logprob_a, pi_net.parameters(),
                    allow_unused=False, create_graph=False, retain_graph=True)
                grad_logpi = torch.vstack(grad_logpi).squeeze(dim=u.feature_dimth)
                premix_t += Ppi_pwr[s0, s]*PI[s, a]*Q[s, a]*grad_logpi
        premix_t -= grad_g # substract grad_g at each timestep term!
        premix_t_norm = torch.linalg.norm(premix_t, ord=None)

        # check here so that we can assert `premix_t_norm`
        if torch.allclose(Ppi_steady[s0, :], Ppi_pwr[s0, :], rtol=tmix_rtol, atol=tmix_atol):
            tmix = t; log['tmix'] = tmix # specific to s0

            # premix diminishing check at mixing
            assert torch.allclose(premix_t_norm, torch.zeros(1).double(),
                rtol=float(arg['premix_diminishing_rtol']), \
                atol=float(arg['premix_diminishing_atol'])), \
                premix_t_norm.item()

            break
        else:
            premix += premix_t # premix so far
            log['premix_angerr'].append(u.get_angular_err(premix.detach().numpy(), grad_b_np))
            log['premix_normerr'].append(torch.linalg.norm(premix - grad_b, ord=None).item())
            # print(t, 'premix', premix.data)
    assert tmix is not None # ensure env.tmax_xep is long enough, bigger than tmix
    # print('tmix', tmix)

    # postmix part
    postmix = torch.zeros(n_param).double()
    for s in range(nS):
        for a in range(nA_list[s]):
            grad_qsa = torch.autograd.grad(Q[s, a], pi_net.parameters(),
                allow_unused=False, create_graph=False, retain_graph=True)
            grad_qsa = torch.vstack(grad_qsa).squeeze()
            postmix += Ppi_steady[s0, s]*PI[s, a]*grad_qsa
    postmix += grad_g # involving: plus grad g!
    log['postmix_normerr'] = torch.linalg.norm(postmix - grad_b, ord=None).item()
    log['postmix_angerr'] = u.get_angular_err(postmix.detach().numpy(), grad_b_np)

    # total
    prepostmix = premix + postmix
    prepostmix_normerr = torch.linalg.norm(prepostmix - grad_b, ord=None)
    prepostmix_angerr = u.get_angular_err(prepostmix.detach().numpy(), grad_b_np)
    log['prepostmix_angerr'] = prepostmix_angerr
    log['prepostmix_normerr'] = prepostmix_normerr.item()
    assert torch.isfinite(prepostmix).all()
    # print('postmix', postmix.data)
    # print('prepostmix', prepostmix.data)
    # print('grad_b', grad_b.data)

    assert torch.allclose(grad_b, prepostmix,
        rtol=float(arg['gradbias_cmp_rtol']), atol=float(arg['gradbias_cmp_atol'])), \
        prepostmix_normerr.data
    return dict(log)

def parse_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', help='cfg filepath', type=str, default=None, required=True)
    parser.add_argument('--ncpu', help='number of cpu', type=int, default=None, required=True)
    parser.add_argument('--repo', help='repo dirpath list', type=str, nargs='+', default=None, required=True)
    parser.add_argument('--custom', help='list of (key: value) items', type=str, nargs='+', default=[])
    arg = parser.parse_args()
    arg.cfg = arg.cfg.replace('file://','')
    return arg

if __name__ == '__main__':
    main()
