#!/usr/bin/env python3
import torch, random, os, pickle, argparse
import numpy as np, pathos.multiprocessing as mp
import gym_util.common_util as gym_u
import polnet as pnet_u, util_bwopt as u
from collections import defaultdict
from poleval_pytorch import get_rpi_s, get_Ppi_ss, get_ppisteady_s, get_Qsa

def main():
    arg = parse_arg(); cfg = vars(arg)
    cfg['ij'] = None # dummy ij coord, only used when making mesh
    maximize_bias_sampling(cfg)

def maximize_bias_sampling(cfg):
    envid = cfg['env']; envid_short = u.get_shortenvid(envid); log = defaultdict(list)
    seed = cfg['seed']; torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    env = gym_u.make_single_env(envid, seed)
    nS, nA = env.nS, env.nA; nA_list = env.nA_list
    Psas = torch.tensor(env.get_Psas()).double()
    Rsa = torch.tensor(env.get_Rsa()).double()
    allstatefeature = torch.from_numpy(env.get_allstatefeature(cfg['sfx']))
    tmix_rtol = env.tmix_cfg['rtol']; tmix_atol = env.tmix_cfg['atol']
    s0_deterministic = env.reset() # assume s0 is deterministic
    tmax_xep = env.tmax_xep if cfg['txep'] is None else cfg['txep']

    PolicyNetwork = pnet_u.policynetclass_dict[cfg['pnet']]
    pi_net = PolicyNetwork(nA_list); pi_net.double()
    n_param = sum([i.numel() for i in pi_net.parameters()])
    init_param = {pi_net.weight_x_name: float(cfg['par'][0]),
        pi_net.weight_y_name: float(cfg['par'][1])}
    for n, p in pi_net.named_parameters():
        p.data.fill_(init_param[n])
        p.data = p.data.double()

    if cfg['cond']=='fisherbias':
        tabs_hat = 2 # aka fisher_transient_withsteadymul_upto_t1
    elif cfg['cond']=='identity':
        tabs_hat = 0 # disable Fisher
    else:
        raise NotImplementedError(cfg['cond'])

    for i in range(cfg['niter']): # optimization (param update) iteration
        # policy evaluation
        PI = pnet_u.policy_net2tabular(allstatefeature, pi_net, requires_grad=True)
        rpi = get_rpi_s(Rsa, PI); Ppi = get_Ppi_ss(Psas, PI)
        ppi_steady = get_ppisteady_s(Ppi, PI)
        Ppi_steady = torch.vstack([ppi_steady]*nS) # unichain: same rows
        Zpi = torch.inverse(torch.eye(nS) - Ppi + Ppi_steady) # fundamental matrix
        Hpi = torch.matmul(Zpi, torch.eye(nS) - Ppi_steady) # deviation matrix
        g = torch.dot(ppi_steady, rpi) # gain
        b = torch.matmul(Hpi, rpi) # bias
        Q = get_Qsa(g, b, Psas, Rsa, nA_list) # action value
        grad_g = torch.autograd.grad(g, pi_net.parameters(),
            allow_unused=False, create_graph=False, retain_graph=True)
        grad_g = torch.vstack(grad_g).squeeze()
        grad_b = torch.autograd.grad(b[s0_deterministic], pi_net.parameters(),
            allow_unused=False, create_graph=False, retain_graph=True)
        grad_b = torch.vstack(grad_b).squeeze()
        log['bias'].append(b[s0_deterministic].item())

        grad_b_hat_list = []; fisher_b_hat_list = []
        for xep_i in range(cfg['mbs']): # a sample corresponds to an xprmt episode
            grad_b_hat = torch.zeros(n_param).double()
            fisher_b_hat = torch.zeros(n_param, n_param).double()
            s0 = env.reset(); s = s0; tmix = None
            log['xep_rew'].append([]) # an empty list for this xprmt episode's rewards

            for t in range(tmax_xep):
                pi = pi_net(torch.from_numpy(env.get_statefeature([s], cfg['sfx'])))
                a = pi.sample().item()
                snext, rnext_val = env.step(a)
                log['xep_rew'][-1].append(rnext_val)
                if cfg['print']:
                    print('{}: t {}, s {}, a {}, snext {}, rew {}, b {:.3f}'.format(
                        xep_i, t, s, a, snext, rnext_val, b[s0]))

                logprob_a = pi.log_prob(torch.tensor([a])) # given the current state s
                grad_logpi = torch.autograd.grad(logprob_a, pi_net.parameters(),
                    allow_unused=False, create_graph=False, retain_graph=True)
                grad_logpi = torch.vstack(grad_logpi).squeeze()

                Ppi_pwr = torch.matrix_power(Ppi, t)
                if torch.allclose(Ppi_steady[s0, :], Ppi_pwr[s0, :],
                    rtol=tmix_rtol, atol=tmix_atol):
                    # compute the final part of bias grad (after mixing)
                    grad_qsa = torch.autograd.grad(Q[s, a], pi_net.parameters(),
                        allow_unused=False, create_graph=False, retain_graph=True)
                    grad_qsa = torch.vstack(grad_qsa).squeeze()
                    grad_b_hat += (grad_qsa + grad_g) # last addition for this xep
                    grad_b_hat_list.append(grad_b_hat)

                    # steady part of the Fisher
                    fisher_b_hat += tabs_hat*torch.outer(grad_logpi, grad_logpi)
                    fisher_b_hat_list.append(fisher_b_hat)

                    tmix = t # specific to s0
                    break # from this xep since the process is already mixing
                else:
                    grad_b_hat += ((Q[s, a]*grad_logpi) - grad_g)

                    if t < tabs_hat:
                        fisher_b_hat += torch.outer(grad_logpi, grad_logpi)

                    s = snext # for next timestep
            assert tmix is not None, 'ij {} tmax_xep {}'.format(cfg['ij'], tmax_xep)

        # stepdir
        if cfg['cond']=='identity':
            C_samplemean = torch.eye(n_param).double()
        elif 'fisher' in cfg['cond']:
            C_samplemean = torch.mean(torch.stack(fisher_b_hat_list), dim=u.sample_dimth)
        else:
            raise NotImplementedError

        grad_b_samplemean = torch.mean(torch.stack(grad_b_hat_list), dim=u.sample_dimth)
        stepdir = torch.matmul(torch.pinverse(C_samplemean), grad_b_samplemean)

        # steplen
        def fn_for_linesearch(param, compute_derivative):
            assert compute_derivative==False # _4ls: for linesearch
            pi_net_4ls = PolicyNetwork(nA_list); pi_net_4ls.double()
            param_dict_4ls = {pi_net_4ls.weight_x_name: param[0], pi_net_4ls.weight_y_name: param[1]}
            for n, p in pi_net_4ls.named_parameters():
                p.data.fill_(param_dict_4ls[n])
                p.data = p.data.double()
            PI_4ls = pnet_u.policy_net2tabular(allstatefeature, pi_net_4ls, requires_grad=False)
            rpi_4ls = get_rpi_s(Rsa, PI_4ls)
            Ppi_4ls = get_Ppi_ss(Psas, PI_4ls)
            ppi_steady_4ls = get_ppisteady_s(Ppi_4ls, PI_4ls)
            Ppi_steady_4ls = torch.vstack([ppi_steady_4ls]*nS) # unichain: same rows
            Zpi_4ls = torch.inverse(torch.eye(nS) - Ppi_4ls + Ppi_steady_4ls) # fundamental matrix
            Hpi_4ls = torch.matmul(Zpi_4ls, torch.eye(nS) - Ppi_steady_4ls) # deviation matrix
            b_4ls = torch.matmul(Hpi_4ls, rpi_4ls) # bias
            return b_4ls[s0_deterministic]

        steplen = u.backtracking_linesearch_ascent(
            p=[p.detach().clone().item() for p in pi_net.parameters()],
            fval=b[s0_deterministic].item(), stepdir=stepdir.detach().clone().numpy(),
            grad=grad_b.detach().clone().numpy(), fn=fn_for_linesearch, niter=100)

        # update param
        log['param'].append([])
        for pidx, p in enumerate(pi_net.parameters()):
            p.data += steplen*stepdir[pidx]
            log['param'][-1].append(p.detach().clone().item())

    # Closure
    initpar_str = '_'.join([str(p) for p in cfg['par']])
    fname = '__'.join(['traj_biasopt_sampling', initpar_str, envid_short])
    logdir = os.path.join(cfg['logdir'], 'data',
        '-'.join(['traj', 'biasoptsampling', cfg['cond'], 'niter{}'.format(cfg['niter']),
        'mbs{}'.format(cfg['mbs']), 'seed{}'.format(seed), cfg['pnet'], envid_short]))

    log['cfg'] = cfg
    for key in ['param']:
        if len(log[key]) > 0: # handle when init at a stationary point
            log[key] = np.array(log[key])

    final_info = {'ij': cfg['ij'], 'logdir': logdir}
    for key in ['bias', 'param']:
        if len(log[key]) > 0:
            final_info[key] = log[key][-1]
        else:
            final_info[key] = None

    os.makedirs(logdir, exist_ok=True)
    if cfg['write']:
        fname += '.pkl'
        with open(os.path.join(logdir, fname), 'wb') as f:
            pickle.dump(log, f)
    else:
        fname += '.txt'
        with open(os.path.join(logdir, fname), 'w') as f:
            f.write('') # empty! just for indicating "done"

    return final_info

def parse_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='env id', type=str, default=None, required=True)
    parser.add_argument('--par', help='init param [x, y]', type=float, nargs='+', default=None, required=True)
    parser.add_argument('--pnet', help='policy network mode id', type=str, default=None, required=True)
    parser.add_argument('--sfx', help='state feature extractor id', type=str, default=None, required=True)
    parser.add_argument('--cond', help='preconditioning matrix mode', type=str, default=None, required=True)
    parser.add_argument('--niter', help='max number of any iteration', type=int, default=None, required=True)
    parser.add_argument('--seed', help='rng seed', type=int, default=None, required=True)
    parser.add_argument('--logdir', help='log dir path', type=str, default=None, required=True)
    parser.add_argument('--print', help='msg printing', type=bool, default=True)
    parser.add_argument('--write', help='trajectory writing', type=bool, default=True)
    parser.add_argument('--txep', help='length of xprmt-episode (overwrite env.tmax_xep)', type=int, default=None)
    arg = parser.parse_args()
    arg.logdir = arg.logdir.replace('file://','')
    return arg

if __name__ == '__main__':
    main()
