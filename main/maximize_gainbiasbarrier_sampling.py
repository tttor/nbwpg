#!/usr/bin/env python3
import torch, random, os, pickle, argparse
import numpy as np, pathos.multiprocessing as mp
import gym_util.common_util as gym_u
import polnet as pnet_u, util_bwopt as u
from collections import defaultdict
from poleval_pytorch import get_rpi_s, get_Ppi_ss, get_ppisteady_s, get_Qsa

def main():
    arg = parse_arg(); cfg = vars(arg)
    cfg['ij'] = (0, 0) # dummy ij coord, only used when making mesh
    maximize_gainbiasbarrier_sampling(cfg)

def maximize_gainbiasbarrier_sampling(cfg):
    objdelim = '_' # use in cfg['obj'] to include the barrier params
    checks = ['gainbiasbarrier' in cfg['obj'], cfg['obj'].count(objdelim)==1]
    checks += [len(cfg['par'])==2]
    if not(all(checks)):
        raise RuntimeError('sanity checks: failed!')

    envid = cfg['env']; envid_short = u.get_shortenvid(envid)
    seed = cfg['seed']; torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    tag = ['traj', 'gainbiasbarrier_optsampling', cfg['cond'], 'niter{}'.format(cfg['niter']),
        'mbs{}'.format(cfg['mbs']), 'seed{}'.format(seed), cfg['pnet'], envid_short]
    logdir = os.path.join(cfg['logdir'], u.make_stamp(tag, timestamp=''))
    log = defaultdict(list); os.makedirs(logdir, exist_ok=True)

    env = gym_u.make_single_env(envid, seed)
    nS, nA = env.nS, env.nA; nA_list = env.nA_list
    Psas = torch.tensor(env.get_Psas()).double()
    Rsa = torch.tensor(env.get_Rsa()).double()
    allstatefeature = torch.from_numpy(env.get_allstatefeature(cfg['sfx']))
    tmix_rtol = env.tmix_cfg['rtol']; tmix_atol = env.tmix_cfg['atol']
    s0_det = env.reset() # assume s0 is deterministic

    PolicyNetwork = pnet_u.policynetclass_dict[cfg['pnet']]
    pi_net = PolicyNetwork(nA_list); pi_net.double()
    nparam = sum([i.numel() for i in pi_net.parameters()])
    init_param = {pi_net.weight_x_name: float(cfg['par'][0]),
        pi_net.weight_y_name: float(cfg['par'][1])}
    for n, p in pi_net.named_parameters():
        p.data.fill_(init_param[n])
        p.data = p.data.double()

    conditioner_mode=cfg['cond']; eps = float(cfg['eps'])
    gainopt = {'converged': False} # approximately converged wrt `eps`
    if 'fisherbias' in conditioner_mode:
        tabs_hat = 2 # fisher_transient_withsteadymul_upto_t1
    else:
        raise NotImplementedError(conditioner_mode)

    for iterx_idx in range(cfg['niterx']): # outer optim iter
        for iter_idx in range(cfg['niter']): # inner optim iter
            fval_mode = 'biasbarrier' if gainopt['converged'] else 'gain'

            # Policy evaluation: exact
            PI = pnet_u.policy_net2tabular(allstatefeature, pi_net, requires_grad=True)
            rpi = get_rpi_s(Rsa, PI); Ppi = get_Ppi_ss(Psas, PI)
            ppi_steady = get_ppisteady_s(Ppi, PI)
            Ppi_steady = torch.vstack([ppi_steady]*nS) # unichain: same rows
            Zpi = torch.inverse(torch.eye(nS) - Ppi + Ppi_steady) # fundamental matrix
            Hpi = torch.matmul(Zpi, torch.eye(nS) - Ppi_steady) # deviation matrix
            g = torch.dot(ppi_steady, rpi) # gain
            b = torch.matmul(Hpi, rpi) # bias
            Q = get_Qsa(g, b, Psas, Rsa, nA_list) # the bias action-value
            if fval_mode=='gain':
                fval = g
            elif fval_mode=='biasbarrier':
                slack = 1
                fval = b[s0_det] + bar*torch.log(g - gainopt['fval'] + slack)
            else:
                raise NotImplementedError(fval_mode)
            grad = torch.autograd.grad(fval, pi_net.parameters(),
                allow_unused=False, create_graph=False, retain_graph=True)
            grad = torch.vstack(grad).squeeze()

            if 'fisher' in conditioner_mode:
                fisher_atallstate = {s: torch.zeros(nparam, nparam).double() for s in range(nS)}
                for s in range(nS):
                    for a in range(nA_list[s]):
                        # Do NOT use torch.log(PI[s, a]): unstable grad on extreme values (with sigmoid fn)
                        pi = pi_net(torch.from_numpy(env.get_statefeature([s], cfg['sfx'])))
                        prob_a = pi.probs.squeeze(dim=u.sample_dimth)[a]
                        logprob_a = pi.log_prob(torch.tensor([a]))
                        grad_logpi = torch.autograd.grad(logprob_a, pi_net.parameters(),
                            allow_unused=False, create_graph=False, retain_graph=True)
                        grad_logpi = torch.vstack(grad_logpi).squeeze()
                        fisher_atallstate[s] += prob_a*torch.outer(grad_logpi, grad_logpi)

                fisher = torch.zeros(nparam, nparam).double()
                if fval_mode=='gain':
                    for s in range(nS):
                        fisher += Ppi_steady[s0_det, s]*fisher_atallstate[s]
                elif fval_mode=='biasbarrier':
                    fisher = None
                    # # This is the `fisher_transient_withsteadymul_upto_t1`
                    # t_begin = 0; t_end = env.tmax_xep; t_transient_max = 1
                    # rtol = env.tmix_cfg['rtol']; atol = env.tmix_cfg['atol']
                    # for t in range(t_begin, t_end + 1, 1):
                    #     Ppi_pwr = torch.matrix_power(Ppi, t)
                    #     if (t <= t_transient_max):
                    #         for s in range(nS):
                    #             fisher += Ppi_pwr[s0_det, s]*fisher_atallstate[s]
                    #     if torch.allclose(Ppi_steady[s0_det,:], Ppi_pwr[s0_det,:], rtol=rtol, atol=atol):
                    #         tau = (t_transient_max + 1) # +1: index gins at 0
                    #         for s in range(nS):
                    #             fisher += (tau*Ppi_steady[s0_det, s])*fisher_atallstate[s]
                    #         break
                    # assert t < env.tmax_xep

                    # # Barrier fisher
                    # grad_g = torch.autograd.grad(g, pi_net.parameters(),
                    #     allow_unused=False, create_graph=False, retain_graph=True)
                    # grad_g = torch.vstack(grad_g).squeeze()
                    # fisher_g = torch.zeros(nparam, nparam).double()
                    # for s in range(nS):
                    #     fisher_g += Ppi_steady[s0, s]*fisher_atallstate[s]
                    # g_delta = (g - gainopt['fval'] + slack)
                    # fisher += bar/g_delta*fisher_g
                    # fisher += bar/(g_delta**2)*torch.outer(grad_g, grad_g) # positive semidefinite
                else:
                    raise NotImplementedError(fval_mode)

            # Sampling-based approx for grad and Fisher
            grad_g_hat_list = []; fisher_g_hat_list = []
            grad_b_hat_list = []; fisher_b_hat_list = []
            for xep_idx in range(cfg['mbs']): # xep: experiment episodes
                grad_g_hat = torch.zeros(nparam).double()
                grad_b_hat = torch.zeros(nparam).double()
                fisher_g_hat = torch.zeros(nparam, nparam).double()
                fisher_b_hat = torch.zeros(nparam, nparam).double()
                s0 = env.reset(); s = s0; tmix = None

                for t in range(env.tmax_xep + 1):
                    pi = pi_net(torch.from_numpy(env.get_statefeature([s], cfg['sfx'])))
                    a = pi.sample().item()
                    snext, _ = env.step(a)

                    logprob_a = pi.log_prob(torch.tensor([a])) # given the current state s
                    grad_logpi = torch.autograd.grad(logprob_a, pi_net.parameters(),
                        allow_unused=False, create_graph=False, retain_graph=True)
                    grad_logpi = torch.vstack(grad_logpi).squeeze()
                    z = Q[s, a]*grad_logpi

                    if cfg['mixing_at_end_of_xep']:
                        mixing = (t==env.tmax_xep)
                    else:
                        Ppi_pwr = torch.matrix_power(Ppi, t)
                        mixing = torch.allclose(Ppi_steady[s0, :], Ppi_pwr[s0, :],
                            rtol=tmix_rtol, atol=tmix_atol)

                    if mixing:
                        # gain grad and Fisher approx
                        grad_g_hat = z # the grad g estimate
                        fisher_g_hat = torch.outer(grad_logpi, grad_logpi) # gain Fisher approx
                        grad_g_hat_list.append(grad_g_hat)
                        fisher_g_hat_list.append(fisher_g_hat)

                        if gainopt['converged']:
                            # bias grad and Fisher approx (post-mixing part)
                            grad_qsa = torch.autograd.grad(Q[s, a], pi_net.parameters(),
                                allow_unused=False, create_graph=False, retain_graph=True)
                            grad_qsa = torch.vstack(grad_qsa).squeeze()
                            grad_b_hat += grad_qsa - (t - 1)*grad_g_hat # `t` subs and `1` add of grad g
                            grad_b_hat_list.append(grad_b_hat)

                            fisher_b_hat += tabs_hat*torch.outer(grad_logpi, grad_logpi)
                            fisher_b_hat_list.append(fisher_b_hat)

                        if not(cfg['mixing_at_end_of_xep']):
                            tmix = t # specific to s0
                        break
                    else:
                        if gainopt['converged']: # now in biasbarrier optim
                            grad_b_hat += z # substraction of grad g is carried out at the end of xep
                            if t < tabs_hat:
                                fisher_b_hat += torch.outer(grad_logpi, grad_logpi)

                        s = snext # for the next timestep

                if not(cfg['mixing_at_end_of_xep']):
                    assert tmix is not None

            if fval_mode=='gain':
                # grad_samplemean = torch.mean(torch.stack(grad_g_hat_list), dim=u.sample_dimth)
                # fisher_samplemean = torch.mean(torch.stack(fisher_g_hat_list), dim=u.sample_dimth)
                grad_samplemean = grad # for devel/debug purpose
                fisher_samplemean = fisher # for devel/debug purpose
            elif fval_mode=='biasbarrier':
                g_delta = (g - gainopt['fval'] + slack)
                grad_g_samplemean = torch.mean(torch.stack(grad_g_hat_list), dim=u.sample_dimth)
                grad_b_samplemean = torch.mean(torch.stack(grad_b_hat_list), dim=u.sample_dimth)
                grad_samplemean = grad_b_samplemean + bar/g_delta*grad_g_samplemean
                if 'fisherbias' in conditioner_mode:
                    fisher_b_samplemean = torch.mean(torch.stack(fisher_b_hat_list), dim=u.sample_dimth)
                    fisher_samplemean = fisher_b_samplemean
                else:
                    raise NotImplementedError(conditioner_mode)
                if 'fisherbarrier' in conditioner_mode:
                    fisher_g_samplemean = torch.mean(torch.stack(fisher_g_hat_list), dim=u.sample_dimth)
                    fisher_samplemean += bar/g_delta*fisher_g_samplemean
                    fisher_samplemean += bar/(g_delta**2)*torch.outer(grad_g_samplemean, grad_g_samplemean) # positive semidefinite
                # grad_samplemean = grad # for devel/debug purpose
                # fisher_samplemean = fisher # for devel/debug purpose
            else:
                raise NotImplementedError(fval_mode)

            grad_norm = torch.linalg.norm(grad, ord=None)
            grad_samplemean_norm = torch.linalg.norm(grad_samplemean, ord=None)
            if cfg['print']:
                    msgs = ['fval {:.5f}'.format(fval.item()),
                    'grad_norm_hat {:.5f}'.format(grad_samplemean_norm.item()),
                    'grad_norm {:.5f}'.format(grad_norm.item()),
                    'gain {:.5f}'.format(g.item()), 'bias {:.5f}'.format(b[s0_det].item()),
                    'xy ({:.5f}, {:.5f})'.format(*[p.detach().clone().item() for p in pi_net.parameters()])]
                    print('{} {} {} {}: '.format(iterx_idx, iter_idx,
                        fval_mode + ('_{:.1f}'.format(bar) if fval_mode=='biasbarrier' else ''),
                        cfg['cond']) + ' '.join(msgs))
            log['fval'].append(fval.item()); log['grad_norm'].append(grad_norm.item())
            log['grad_samplemean_norm'].append(grad_samplemean_norm.item())
            log['gain'].append(g.item()); log['bias'].append(b[s0_det].item())
            log['param'].append(torch.tensor([p.detach().clone().item() for p in pi_net.parameters()]))

            # Convergence check
            if grad_samplemean_norm < eps: # `less than` for a strictly feasible point
                if gainopt['converged']:
                    pass # continue biasbarrier optim till niter runs out
                    # break # *from* biasbarrier optim
                else:
                    gainopt['converged'] = True
                    gainopt_info = {'ith_iter': iter_idx,
                        'param': [p.detach().clone().item() for p in pi_net.parameters()],
                        'fval': fval.item(), 'grad': grad.detach().clone().numpy(),
                        'grad_norm': grad_norm.item(), 'grad_samplemean_norm': grad_samplemean_norm.item()}
                    gainopt = {**gainopt, **gainopt_info}
                    bar = float(cfg['obj'].split(objdelim)[1])
                    continue # *to* biasbarrier optimization

            # Stepdir
            if 'fisher' in conditioner_mode:
                C_samplemean = fisher_samplemean
            else:
                raise NotImplementedError(conditioner_mode)
            C_samplemean[C_samplemean < 1e-128] = 1e-128 # otherwise, `pinverse` may yield +-inf
            stepdir = torch.matmul(torch.pinverse(C_samplemean), grad_samplemean)
            log['stepdir'].append(stepdir.detach().clone())
            assert torch.isfinite(stepdir).all()

            # Steplen: exact
            def fn_for_linesearch(param, compute_derivative):
                assert compute_derivative==False # _4ls: for linesearch
                pi_net_4ls = PolicyNetwork(nA_list); pi_net_4ls.double()
                param_dict_4ls = {pi_net_4ls.weight_x_name: param[0], pi_net_4ls.weight_y_name: param[1]}
                for n, p in pi_net_4ls.named_parameters():
                    p.data.fill_(param_dict_4ls[n])
                    p.data = p.data.double()
                PI_4ls = pnet_u.policy_net2tabular(allstatefeature, pi_net_4ls, requires_grad=False)
                rpi_4ls = get_rpi_s(Rsa, PI_4ls); Ppi_4ls = get_Ppi_ss(Psas, PI_4ls)
                ppi_steady_4ls = get_ppisteady_s(Ppi_4ls, PI_4ls)
                g_4ls = torch.dot(ppi_steady_4ls, rpi_4ls) # gain
                if fval_mode=='gain':
                    fval_4ls = g_4ls
                elif fval_mode=='biasbarrier':
                    Ppi_steady_4ls = torch.vstack([ppi_steady_4ls]*nS) # unichain: same rows
                    Zpi_4ls = torch.inverse(torch.eye(nS) - Ppi_4ls + Ppi_steady_4ls) # fundamental matrix
                    Hpi_4ls = torch.matmul(Zpi_4ls, torch.eye(nS) - Ppi_steady_4ls) # deviation matrix
                    b_4ls = torch.matmul(Hpi_4ls, rpi_4ls) # bias
                    fval_4ls = b_4ls[s0_det] + bar*torch.log(g_4ls - gainopt['fval'] + slack)
                else:
                    raise NotImplementedError(fval_mode)
                return fval_4ls

            steplen = u.backtracking_linesearch_ascent(
                p=[p.detach().clone().item() for p in pi_net.parameters()],
                fval=fval.item(), grad=grad.detach().clone().numpy(),
                stepdir=stepdir.detach().clone().numpy(),
                fn=fn_for_linesearch, niter=100)
            log['steplen'].append(steplen)

            # Update param
            for pidx, p in enumerate(pi_net.parameters()):
                p.data += steplen*stepdir[pidx]

        # Update the barrier parameter (only for biasbarrier optim)
        if gainopt['converged']:
            bar /= 10

    # Closure
    log = dict(log); log['cfg'] = cfg
    final_info = {'niter': iter_idx+1, 'ij': cfg['ij'], 'logdir': logdir,
        'gainopt_converged': int(gainopt['converged'])}
    final_info['gainopt_fval'] = gainopt['fval'] if ('fval' in gainopt.keys()) else np.nan
    final_info['gainopt_gradnorm'] = gainopt['grad_norm'] if ('grad_norm' in gainopt.keys()) else np.nan
    final_info['gainopt_gradsamplemeannorm'] = gainopt['grad_samplemean_norm'] \
        if ('grad_samplemean_norm' in gainopt.keys()) else np.nan
    for key in ['param', 'stepdir', 'fval', 'gain', 'bias', 'grad_norm', 'grad_samplemean_norm']:
        final_info[key] = log[key][-1] if (len(log[key]) > 0) else None
    final_info['bias_diff'] = final_info['bias'] - env.cfg['deterministic_policy']['bs0max_gainmax']
    final_info['gain_diff'] = final_info['gain'] - env.cfg['deterministic_policy']['gain_max']

    fname = '__'.join(['_'.join([str(idx).zfill(2) for idx in cfg['ij']]),
        'traj_samplingopt', fval_mode, cfg['cond'], envid_short, str(cfg['par']).replace(' ', '')])
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
    parser.add_argument('--obj', help='optimization objective, containing barrier params', type=str, default=None, required=True)
    parser.add_argument('--par', help='init param [x, y]', type=float, nargs='+', default=None, required=True)
    parser.add_argument('--pnet', help='policy network mode id', type=str, default=None, required=True)
    parser.add_argument('--sfx', help='state feature extractor id', type=str, default=None, required=True)
    parser.add_argument('--cond', help='preconditioning matrix mode', type=str, default=None, required=True)
    parser.add_argument('--niter', help='max number of inner optim iteration', type=int, default=None, required=True)
    parser.add_argument('--niterx', help='max number of outer optim iteration', type=int, default=None, required=True)
    parser.add_argument('--mbs', help='minibatch size', type=int, default=None, required=True)
    parser.add_argument('--eps', help='epsilon for stationary points', type=float, default=None, required=True)
    parser.add_argument('--mixing_at_end_of_xep', help='mixing approx time', type=int, default=None, required=True)
    parser.add_argument('--seed', help='rng seed', type=int, default=None, required=True)
    parser.add_argument('--logdir', help='log dir path', type=str, default=None, required=True)
    parser.add_argument('--print', help='msg printing', type=int, default=1)
    parser.add_argument('--write', help='trajectory writing', type=int, default=1)
    arg = parser.parse_args()
    arg.logdir = arg.logdir.replace('file://','')
    return arg

if __name__ == '__main__':
    main()
