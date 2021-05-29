#!/usr/bin/env python3
import torch, random, sys, os, pickle, argparse
import numpy as np, pathos.multiprocessing as mp
import gym_util.common_util as cou
import polnet as pnet, util_bwopt as u
from collections import defaultdict
from poleval_pytorch import get_rpi_s, get_Ppi_ss, get_ppisteady_s

def main():
    arg = parse_arg(); cfg = vars(arg)
    cfg['ij'] = None # dummy ij coord, used in making mesh
    maximize_singleobj_exactly(cfg)

def maximize_singleobj_exactly(cfg):
    def fn(param, compute_derivative=True):
        param_dict = {pi_net.weight_x_name: param[0], pi_net.weight_y_name: param[1]}
        for n, p in pi_net.named_parameters():
            p.data.fill_(param_dict[n])
            p.data = p.data.double()

        PI = pnet.policy_net2tabular(allstatefea, pi_net, requires_grad=True)
        rpi = get_rpi_s(Rsa, PI); Ppi = get_Ppi_ss(Psas, PI)
        ppi_steady = get_ppisteady_s(Ppi, PI)
        Ppi_steady = torch.vstack([ppi_steady]*nS) # unichain: same rows
        I = torch.eye(nS).double()
        Dpi = torch.inverse(I - gamma*Ppi) # IMPROPER discounted state distrib
        Zpi = torch.inverse(I - Ppi + Ppi_steady) # fundamental matrix
        Hpi = torch.matmul(Zpi, I - Ppi_steady) # deviation matrix
        g = torch.dot(ppi_steady, rpi) # gain
        b = torch.matmul(Hpi, rpi) # bias
        d = torch.matmul(Dpi, rpi) # discounted

        if fval_mode=='gain':
            fval = g
        elif fval_mode=='bias':
            fval = b[s0]
        elif fval_mode=='disc':
            fval = (1 - gamma)*d[s0] # the scaled value for sampling-enabler expression
        # elif fval_mode=='pena':
        #     # `create_graph` needs to be `True` so that `grad_g_norm.requires_grad= True`
        #     grad_g = torch.autograd.grad(g, pi_net.parameters(),
        #         allow_unused=False, create_graph=True, retain_graph=True)
        #     grad_g = torch.vstack(grad_g).squeeze()
        #     assert torch.isfinite(grad_g).all()
        #     grad_g_norm = torch.linalg.norm(grad_g, ord=None)
        #     fval = b[s0] - 0.5*pen*(grad_g_norm**2)
        else:
            raise NotImplementedError(fval_mode)

        if compute_derivative:
            grad = torch.autograd.grad(fval, pi_net.parameters(),
                allow_unused=False, create_graph=True, retain_graph=True)
            grad = torch.vstack(grad).squeeze()
            assert torch.isfinite(grad).all()

            hess = []
            for pidx in range(n_param):
                mask = torch.zeros(n_param); mask[pidx] = 1.
                hess_i = torch.autograd.grad(grad, pi_net.parameters(),
                    grad_outputs=mask, allow_unused=False, create_graph=False, retain_graph=True)
                hess.append(torch.vstack(hess_i).squeeze())
            hess = torch.vstack(hess); assert torch.isfinite(hess).all()

            if 'fisher' in conditioner_mode:
                fisher_atallstate = {s: torch.zeros(n_param, n_param).double() for s in range(nS)}
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

                fisher = torch.zeros(n_param, n_param).double()
                rtol = env.tmix_cfg['rtol']; atol = env.tmix_cfg['atol']
                if conditioner_mode=='fisher_steady' and fval_mode=='gain':
                    for s in range(nS):
                        fisher += Ppi_steady[s0, s]*fisher_atallstate[s]
                elif conditioner_mode=='fisher_disc' and fval_mode=='disc':
                    for s in range(nS):
                        fisher += (1 - gamma)*Dpi[s0, s]*fisher_atallstate[s]
                elif conditioner_mode=='fisher_disc_unnormalized' and fval_mode=='disc':
                    for s in range(nS):
                        fisher += Dpi[s0, s]*fisher_atallstate[s]
                elif conditioner_mode=='fisher_devmat' and fval_mode=='bias':
                    for s in range(nS):
                        fisher += torch.abs(Hpi[s0, s])*fisher_atallstate[s]
                elif conditioner_mode=='fisher_probtransient' and fval_mode=='bias':
                    t_begin = 0; t_end = tmax_xep
                    for t in range(t_begin, t_end + 1, 1):
                        Ppi_pwr = torch.matrix_power(Ppi, t)
                        for s in range(nS):
                            pb_s = torch.abs(Ppi_pwr[s0, s] - Ppi_steady[s0, s])
                            fisher += pb_s*fisher_atallstate[s]
                        if torch.allclose(Ppi_steady[s0,:], Ppi_pwr[s0,:], rtol=rtol, atol=atol):
                            break
                    assert t <= tmax_xep
                elif ('fisher_transient_withsteadymul_upto_t' in conditioner_mode) and fval_mode=='bias' :
                    t_begin = 0; t_end = tmax_xep
                    t_transient_max = int(conditioner_mode.replace('fisher_transient_withsteadymul_upto_t', ''))
                    for t in range(t_begin, t_end + 1, 1):
                        Ppi_pwr = torch.matrix_power(Ppi, t)
                        if (t <= t_transient_max): # equiv to `t < tabs`
                            for s in range(nS):
                                fisher += Ppi_pwr[s0, s]*fisher_atallstate[s]
                        if torch.allclose(Ppi_steady[s0,:], Ppi_pwr[s0,:], rtol=rtol, atol=atol):
                            tau = (t_transient_max + 1) # +1: index begins at 0
                            for s in range(nS):
                                fisher += (tau*Ppi_steady[s0, s])*fisher_atallstate[s]
                            break
                    assert t <= tmax_xep
                # elif conditioner_mode=='fisher_pena' and fval_mode=='pena':
                #     # gain part
                #     fisher_g = torch.zeros(n_param, n_param).double()
                #     for s in range(nS):
                #         fisher_g += Ppi_steady[s0, s]*fisher_atallstate[s]
                #     # bias part
                #     t_begin = 0; t_end = tmax_xep; t_transient_max = 1
                #     fisher_b = torch.zeros(n_param, n_param).double()
                #     for t in range(t_begin, t_end + 1, 1):
                #         Ppi_pwr = torch.matrix_power(Ppi, t)
                #         if (t <= t_transient_max):
                #             for s in range(nS):
                #                 fisher_b += Ppi_pwr[s0, s]*fisher_atallstate[s]
                #         if torch.allclose(Ppi_steady[s0,:], Ppi_pwr[s0,:], rtol=rtol, atol=atol):
                #             tau = (t_transient_max + 1) # +1: index gins at 0
                #             for s in range(nS):
                #                 fisher_b += (tau*Ppi_steady[s0, s])*fisher_atallstate[s]
                #             break
                #     assert t <= tmax_xep
                #     # fisher for the penalized obj
                #     fisher = fisher_b - pen*fisher_g
                else:
                    raise NotImplementedError(conditioner_mode, fval_mode)
                cond = fisher
            elif conditioner_mode=='hess':
                def modify_hess_to_negativedefinite(H):
                    kappa_min = 1e-2; kappa_multiplier = 2; kappa = 0.
                    for k in range(100):
                        try:
                            H_mod = H - kappa*torch.eye(n_param)
                            torch.cholesky(-H_mod) # test for positive definiteness
                            return H_mod
                        except:
                            # print('{} cholesky: failed using kappa {:.3f}'.format(k, kappa))
                            kappa = max(kappa_multiplier*kappa, kappa_min)
                    raise RuntimeError
                cond = -1.*modify_hess_to_negativedefinite(hess)
            elif conditioner_mode=='identity':
                cond = torch.eye(n_param).double()
            else:
                raise NotImplementedError

            assert torch.isfinite(cond).all()
            return (fval, grad, hess, cond, {'gain': g, 'bias': b[s0], 'disc': d[s0]})
        else:
            return fval
    ############################################################# fn: end ######

    envid = cfg['env']; seed = cfg['seed']; envid_short = u.get_shortenvid(envid)
    tag = ['traj_exactopt', cfg['obj'], cfg['cond'], cfg['pnet'], envid_short]
    logdir = os.path.join(cfg['logdir'], u.make_stamp(tag, timestamp=''))
    log = defaultdict(list); os.makedirs(logdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed) # not used in exact

    env = cou.make_single_env(envid, seed)
    Psas = torch.tensor(env.get_Psas()).double()
    Rsa = torch.tensor(env.get_Rsa()).double()
    allstatefea = torch.from_numpy(env.get_allstatefeature(cfg['sfx']))
    s0 = env.reset(); nS, nA = env.nS, env.nA; nA_list = env.nA_list
    tmax_xep = env.tmax_xep if cfg['txep'] is None else cfg['txep']

    PolicyNetClass = pnet.policynetclass_dict[cfg['pnet']]
    pi_net = PolicyNetClass(nA_list)
    n_param = sum([i.numel() for i in pi_net.parameters()])

    p = torch.tensor(cfg['par']); assert len(cfg['par'])==2
    conditioner_mode=cfg['cond']; eps = float(cfg['eps'])
    fval_mode = cfg['obj'][0:4] # 4 letters: gain, bias, disc (eg `disc_0.99`)
    gamma = float(cfg['obj'][5:]) if fval_mode=='disc' else float('Nan') # discount factor
    pen, pen_mul = [float(i) for i in cfg['obj'][5:].split('_')] \
        if fval_mode=='pena' else [float('Nan')]*2 # objective with penalty

    for iterx_idx in range(cfg['niterx']): # outer optim loop
        for iter_idx in range(cfg['niter']): # inner optim loop
            # Evaluate
            fval, grad, hess, cond, info = fn(p)
            grad_norm = torch.linalg.norm(grad)
            hess_eigval = torch.eig(hess, eigenvectors=False).eigenvalues[:, 0] # real part @idx=0

            log['param'].append(p.detach().clone())
            log['fval'].append(fval.item()); log['grad_norm'].append(grad_norm.item())
            log['gain'].append(info['gain'].item()); log['bias'].append(info['bias'].item())
            log['disc'].append(info['disc'].item()); log['pen'].append(pen)
            if cfg['print']:
                msgs = ['fval {:.5f}'.format(fval.item()), 'grad_norm {:.5f}'.format(grad_norm.item()),
                    'hess_eigval ({:.5f}, {:.5f})'.format(hess_eigval[0], hess_eigval[1]),
                    'gain {:.5f}'.format(info['gain'].item()), 'bias {:.5f}'.format(info['bias'].item()),
                    'disc {:.5f}'.format(info['disc'].item()), 'pen {:.5f}'.format(pen),
                    'xy ({:.5f}, {:.5f})'.format(p[0], p[1])]
                print('{} {} {} {}: '.format(iterx_idx, iter_idx, fval_mode, cfg['cond']) + ' '.join(msgs))

            if (grad_norm <= eps) and (hess_eigval <= 0.).all():
                break

            # Step (ascent) direction
            # Using psuedo inverse, accomodating zero fisher_steady on bias-optim
            stepdir = torch.matmul(torch.pinverse(cond), grad)
            log['stepdir'].append(stepdir.detach().clone())

            # Step size
            steplen = u.backtracking_linesearch_ascent(
                p=p.detach().clone().numpy(),
                fval=fval.item(), grad=grad.detach().clone().numpy(),
                stepdir=stepdir.detach().clone().numpy(),
                fn=fn, niter=100)
            log['steplen'].append(steplen)

            # Update parameters
            p += steplen*stepdir

        # update penalty param (outer optim loop)
        pen = pen*pen_mul

    # Closure
    log = dict(log); log['cfg'] = cfg
    final_info = {'niter': iter_idx+1, 'ij': cfg['ij'], 'logdir': logdir, 'discountfactor': gamma}
    for key in ['param', 'fval', 'grad_norm', 'gain', 'bias', 'disc']:
        if len(log[key]) > 0:
            final_info[key] = log[key][-1]
        else:
            final_info[key] = None
    final_info['bias_diff'] = final_info['bias'] - env.cfg['deterministic_policy']['bs0max_gainmax']
    final_info['gain_diff'] = final_info['gain'] - env.cfg['deterministic_policy']['gain_max']

    initpar_str = '_'.join([str(p) for p in cfg['par']])
    fname = '__'.join(['traj_exactopt', fval_mode, cfg['cond'], initpar_str, envid_short])
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
    parser.add_argument('--obj', help='optimization objective', type=str, default=None, required=True)
    parser.add_argument('--par', help='init param [x, y]', type=float, nargs='+', default=None, required=True)
    parser.add_argument('--pnet', help='policy network mode id', type=str, default=None, required=True)
    parser.add_argument('--sfx', help='state feature extractor id', type=str, default=None, required=True)
    parser.add_argument('--eps', help='epsilon for grad norm', type=float, default=None, required=True)
    parser.add_argument('--cond', help='preconditioning matrix mode', type=str, default=None, required=True)
    parser.add_argument('--niter', help='max number of any inner optim iteration', type=int, default=None, required=True)
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
