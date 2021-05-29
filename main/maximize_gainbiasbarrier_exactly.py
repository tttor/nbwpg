#!/usr/bin/env python3
import torch, random, sys, os, pickle, argparse
import numpy as np, pathos.multiprocessing as mp
import gym_util.common_util as cou
import polnet as pnet, util_bwopt as u
from collections import defaultdict
from poleval_pytorch import get_rpi_s, get_Ppi_ss, get_ppisteady_s

def main():
    arg = parse_arg(); cfg = vars(arg)
    cfg['ij'] = (0, 0) # dummy ij coord, used in making mesh
    maximize_gainbiasbarrier_exactly(cfg)

def maximize_gainbiasbarrier_exactly(cfg):
    def fn(param, compute_derivative=True):
        param_dict = {pi_net.weight_x_name: param[0], pi_net.weight_y_name: param[1]}
        for n, p in pi_net.named_parameters():
            p.data.fill_(param_dict[n])
            p.data = p.data.double()

        PI = pnet.policy_net2tabular(allstatefea, pi_net, requires_grad=True)
        rpi = get_rpi_s(Rsa, PI); Ppi = get_Ppi_ss(Psas, PI)
        ppi_steady = get_ppisteady_s(Ppi, PI)
        Ppi_steady = torch.vstack([ppi_steady]*nS) # unichain: same rows
        Zpi = torch.inverse(torch.eye(nS) - Ppi + Ppi_steady) # fundamental matrix
        Hpi = torch.matmul(Zpi, torch.eye(nS) - Ppi_steady) # deviation matrix
        g = torch.dot(ppi_steady, rpi) # gain
        b = torch.matmul(Hpi, rpi) # bias

        if fval_mode=='gain':
            fval = g
        elif fval_mode=='biasbarrier':
            g_delta = (g - gainopt['fval'] + 1)
            fval = b[s0] + bar*torch.log(g_delta)
        else:
            raise NotImplementedError(fval_mode)

        if compute_derivative:
            grad = torch.autograd.grad(fval, pi_net.parameters(),
                allow_unused=False, create_graph=True, retain_graph=True)
            grad = torch.vstack(grad).squeeze(dim=u.feature_dimth)
            assert torch.isfinite(grad).all()

            hess = []
            for pidx in range(nparam):
                mask = torch.zeros(nparam); mask[pidx] = 1.
                hess_i = torch.autograd.grad(grad, pi_net.parameters(),
                    grad_outputs=mask, allow_unused=False, create_graph=False, retain_graph=True)
                hess.append(torch.vstack(hess_i).squeeze(dim=u.feature_dimth))
            hess = torch.vstack(hess); assert torch.isfinite(hess).all()

            if conditioner_mode=='fisher':
                fisher_atallstate = {s: torch.zeros(nparam, nparam).double() for s in range(nS)}
                for s in range(nS):
                    for a in range(nA_list[s]):
                        # Do NOT use torch.log(PI[s, a]): unstable grad on extreme values (with sigmoid fn)
                        pi = pi_net(torch.from_numpy(env.get_statefeature([s], cfg['sfx'])))
                        prob_a = pi.probs.squeeze(dim=u.sample_dimth)[a]
                        logprob_a = pi.log_prob(torch.tensor([a]))
                        grad_logpi = torch.autograd.grad(logprob_a, pi_net.parameters(),
                            allow_unused=False, create_graph=False, retain_graph=True)
                        grad_logpi = torch.vstack(grad_logpi).squeeze(dim=u.feature_dimth)
                        fisher_atallstate[s] += prob_a*torch.outer(grad_logpi, grad_logpi)

                fisher = torch.zeros(nparam, nparam).double()
                if fval_mode=='gain':
                    for s in range(nS):
                        fisher += Ppi_steady[s0, s]*fisher_atallstate[s]
                elif fval_mode=='biasbarrier':
                    # This is the `fisher_transient_withsteadymul_upto_t1`
                    t_begin = 0; t_end = env.tmax_xep; t_transient_max = 1
                    rtol = env.tmix_cfg['rtol']; atol = env.tmix_cfg['atol']
                    for t in range(t_begin, t_end + 1, 1):
                        Ppi_pwr = torch.matrix_power(Ppi, t)
                        if (t <= t_transient_max):
                            for s in range(nS):
                                fisher += Ppi_pwr[s0, s]*fisher_atallstate[s]
                        if torch.allclose(Ppi_steady[s0,:], Ppi_pwr[s0,:], rtol=rtol, atol=atol):
                            tau = (t_transient_max + 1) # +1: index gins at 0
                            for s in range(nS):
                                fisher += (tau*Ppi_steady[s0, s])*fisher_atallstate[s]
                            break
                    assert t < env.tmax_xep

                    # Barrier fisher-like matrix
                    grad_g = torch.autograd.grad(g, pi_net.parameters(),
                        allow_unused=False, create_graph=True, retain_graph=True)
                    grad_g = torch.vstack(grad_g).squeeze()
                    assert torch.isfinite(grad_g).all()

                    fisher_g = torch.zeros(nparam, nparam).double()
                    for s in range(nS):
                        fisher_g += Ppi_steady[s0, s]*fisher_atallstate[s]
                    fisher += bar/g_delta*fisher_g
                    fisher += bar/(g_delta**2)*torch.outer(grad_g, grad_g) # positive semidefinite
                else:
                    raise NotImplementedError(fval_mode)
                cond = fisher
            elif conditioner_mode=='identity':
                cond = torch.eye(nparam).double()
            else:
                raise NotImplementedError(conditioner_mode)

            assert torch.isfinite(cond).all()
            return (fval, grad, hess, cond, {'gain': g, 'bias': b[s0]})
        else:
            return fval

    ############################################################################
    objdelim = '_' # use in cfg['obj'] to include the barrier params
    checks = ['gainbiasbarrier' in cfg['obj'], cfg['obj'].count(objdelim)==1]
    checks += [len(cfg['par'])==2]
    if not(all(checks)):
        raise RuntimeError('sanity checks: failed!')

    envid = cfg['env']; seed = cfg['seed']; envid_short = u.get_shortenvid(envid)
    tag = ['traj', 'gainbiasbarrieroptexact', cfg['cond'], cfg['pnet'], envid_short]
    logdir = os.path.join(cfg['logdir'], u.make_stamp(tag, timestamp=''))
    log = defaultdict(list); os.makedirs(logdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    env = cou.make_single_env(envid, seed)
    Psas = torch.tensor(env.get_Psas()).double()
    Rsa = torch.tensor(env.get_Rsa()).double()
    allstatefea = torch.from_numpy(env.get_allstatefeature(cfg['sfx']))
    s0 = env.reset(); nS, nA = env.nS, env.nA; nA_list = env.nA_list

    PolicyNetClass = pnet.policynetclass_dict[cfg['pnet']]
    pi_net = PolicyNetClass(nA_list); pi_net.double()
    nparam = sum([i.numel() for i in pi_net.parameters()])

    p = torch.tensor(cfg['par'])
    conditioner_mode = cfg['cond']; eps = float(cfg['eps'])
    gainopt = {'converged': False} # approximately converged wrt `eps`

    for iterx_idx in range(cfg['niterx']): # outer optim iter
        for iter_idx in range(cfg['niter']): # inner optim iter
            # Evaluate
            fval_mode = 'biasbarrier' if gainopt['converged'] else 'gain'
            fval, grad, hess, cond, info = fn(p)
            grad_norm = torch.linalg.norm(grad)
            hess_eigval = torch.eig(hess, eigenvectors=False).eigenvalues[:, 0] # real part @idx=0

            log['param'].append(p.detach().clone())
            log['fval'].append(fval.item()); log['grad_norm'].append(grad_norm.item())
            log['gain'].append(info['gain'].item()); log['bias'].append(info['bias'].item())
            if cfg['print']:
                msgs = ['fval {:.5f}'.format(fval.item()), 'grad_norm {:.5f}'.format(grad_norm.item()),
                    'hess_eigval ({:.5f}, {:.5f})'.format(hess_eigval[0], hess_eigval[1]),
                    'gain {:.5f}'.format(info['gain'].item()), 'bias {:.5f}'.format(info['bias'].item()),
                    'xy ({:.5f}, {:.5f})'.format(p[0], p[1])]
                print('{} {} {} {}: '.format(iterx_idx, iter_idx,
                    fval_mode + ('_{:.1f}'.format(bar) if fval_mode=='biasbarrier' else ''),
                    cfg['cond']) + ' '.join(msgs))

            # convergence check, determining when to switch the optim obj
            # if (grad_norm <= eps) and (hess_eigval <= 0.).all():
            if (grad_norm < eps):
                if gainopt['converged']:
                    pass # continue till niterx runs out, in theory as k to infty
                    # break # *from* the current k-th biasbarrier optimization
                else:
                    gainopt['converged'] = True
                    gainopt_info = {'param': p.detach().clone(), 'ith_iter': iter_idx,
                        'fval': fval.item(), 'grad': grad.detach().clone(),
                        'grad_norm': grad_norm.item()}
                    gainopt = {**gainopt, **gainopt_info}
                    bar = float(cfg['obj'].split(objdelim)[1])
                    continue # *to* biasbarrier optimization

            # Step (ascent) direction (Using pseudo inverse)
            cond[cond < 1e-128] = 1e-128 # otherwise, `pinverse` may yield +-inf
            stepdir = torch.matmul(torch.pinverse(cond), grad)
            log['stepdir'].append(stepdir.detach().clone())
            assert torch.isfinite(stepdir).all()

            # Step size (step length)
            steplen = u.backtracking_linesearch_ascent(
                p=p.detach().clone().numpy(),
                fval=fval.item(), grad=grad.detach().clone().numpy(),
                stepdir=stepdir.detach().clone().numpy(),
                fn=fn, niter=100)
            log['steplen'].append(steplen)

            # Update parameters
            p += steplen*stepdir

        # Update the barrier parameter (only for biasbarrier optim)
        if gainopt['converged']:
            bar /= 10

    # Closure
    log = dict(log); log['cfg'] = cfg
    final_info = {'niter': iter_idx+1, 'ij': cfg['ij'], 'logdir': logdir,
        'gainopt_converged': int(gainopt['converged'])}
    final_info['gainopt_fval'] = gainopt['fval'] if ('fval' in gainopt.keys()) else np.nan
    final_info['gainopt_gradnorm'] = gainopt['grad_norm'] if ('grad_norm' in gainopt.keys()) else np.nan
    for key in ['param', 'stepdir', 'fval', 'grad_norm', 'gain', 'bias']:
        final_info[key] = log[key][-1] if (len(log[key]) > 0) else None
    final_info['bias_diff'] = final_info['bias'] - env.cfg['deterministic_policy']['bs0max_gainmax']
    final_info['gain_diff'] = final_info['gain'] - env.cfg['deterministic_policy']['gain_max']

    fname = '__'.join(['_'.join([str(idx).zfill(2) for idx in cfg['ij']]),
        'traj_exactopt', fval_mode, cfg['cond'], envid_short, str(cfg['par']).replace(' ', '')])
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
    parser.add_argument('--eps', help='epsilon for grad norm for convergence check', type=float, default=None, required=True)
    parser.add_argument('--cond', help='preconditioning matrix mode', type=str, default=None, required=True)
    parser.add_argument('--niter', help='max number of inner optim iteration', type=int, default=None, required=True)
    parser.add_argument('--niterx', help='max number of outer optim iteration', type=int, default=None, required=True)
    parser.add_argument('--seed', help='rng seed', type=int, default=None, required=True)
    parser.add_argument('--logdir', help='log dir path', type=str, default=None, required=True)
    parser.add_argument('--print', help='msg printing', type=bool, default=True)
    parser.add_argument('--write', help='trajectory writing', type=bool, default=True)
    arg = parser.parse_args()
    arg.logdir = arg.logdir.replace('file://','')
    return arg

if __name__ == '__main__':
    main()
