#!/usr/bin/env python3
import os, yaml, argparse, pickle, shutil
import numpy as np, pathos.multiprocessing as mp
import util_bwopt as u
from collections import defaultdict
from maximize_bias_sampling import maximize_bias_sampling
from maximize_gainbiasbarrier_sampling import maximize_gainbiasbarrier_sampling
from maximize_singleobj_exactly import maximize_singleobj_exactly
from maximize_gainbiasbarrier_exactly import maximize_gainbiasbarrier_exactly

def main():
    arg = parse_arg(); cfg = u.load_cfg(arg.cfg, arg)
    wxmat, wymat = u.get_wxwymesh(cfg)
    datadir = os.path.join(cfg['logdir'], 'data'); os.makedirs(datadir, exist_ok=True)
    print(cfg); print('wxmat.shape', wxmat.shape)

    maximizer_dict = {
        'sampling': {'bias': maximize_bias_sampling,
                     'gainbiasbarrier': maximize_gainbiasbarrier_sampling},
        'exact': {'gain': maximize_singleobj_exactly, 'bias': maximize_singleobj_exactly,
            'disc': maximize_singleobj_exactly, 'pena': maximize_singleobj_exactly,
            'gainbiasbarrier': maximize_gainbiasbarrier_exactly}
    }
    objstr_delim = '_' # convention: after the first delim is additional info/params
    maximizer = maximizer_dict[arg.opt][arg.obj.split(objstr_delim)[0]]

    for seed_idx, seed in enumerate(cfg['seed']):
        print('optimizing...', seed, seed_idx+1, len(cfg['seed']))
        cfg_common = {'env': cfg['envid'], 'obj': arg.obj, 'cond': cfg['conditioner_mode'],
            'pnet': cfg['polnet']['mode'], 'sfx': cfg['polnet']['state_feature_extractor_id'],
            'niter': cfg['niter'], 'niterx': cfg['niter_outer'],
            'mixing_at_end_of_xep': int(cfg['mixing_at_end_of_xep']),
            'mbs': cfg['minibatch_size'], 'eps': float(cfg['epsilon']), 'seed': seed,
            'logdir': datadir, 'print': bool(cfg['print_msg']), 'write': bool(cfg['write_traj'])}
        if 'tmax_xep' in cfg.keys():
            cfg_common['txep'] = cfg['tmax_xep']
        else:
            cfg_common['txep'] = None

        nrow, ncol = wxmat.shape
        cfg_generator = ({**cfg_common, **{'par': (wxmat[i, j], wymat[i, j]), 'ij': (i, j)}} \
            for i in range(nrow) for j in range(ncol))

        finalinfo_list = []
        if cfg['ncpu']==1:
            for cfg_idx, cfg_i in enumerate(cfg_generator):
                # if not(cfg_idx==27):
                #     continue
                print('cfg_idx', cfg_idx, nrow*ncol, cfg_i['ij'], cfg_i['par'])
                finalinfo = maximizer(cfg_i)
                finalinfo_list.append(finalinfo)
        else:
            pool = mp.ProcessingPool(ncpus=cfg['ncpu'])
            finalinfo_list = pool.map(maximizer, cfg_generator)

        print('making meshdata...')
        meshdata = defaultdict(dict)
        for finalinfo in finalinfo_list:
            for k, v in finalinfo.items():
                if k=='ij':
                    continue
                meshdata[k][finalinfo['ij']] = v
        meshdata = dict(meshdata)
        meshdata['wxmat'] = wxmat; meshdata['wymat'] = wymat
        meshdata['cfg'] = cfg; meshdata['seed'] = seed

        print('writing meshdata...')
        tag = ['meshdata_opt', arg.opt, arg.obj, cfg['conditioner_mode'],
            'niter{}'.format(cfg['niter']), 'mbs{}'.format(cfg['minibatch_size']),
            'eps{}'.format(cfg['epsilon']), 'res{}'.format(cfg['resolution']),
            'seed{}'.format(seed), cfg['polnet']['mode'], u.get_shortenvid(cfg['envid'])]
        fname = u.make_stamp(tag, cfg['timestamp']) + '.pkl'
        with open(os.path.join(datadir, fname), 'wb') as f:
            pickle.dump(meshdata, f)

        print('removing dummy log...')
        shutil.rmtree(finalinfo_list[0]['logdir'])

def parse_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', help='cfg filepath', type=str, default=None, required=True)
    parser.add_argument('--ncpu', help='number of cpu', type=int, default=None, required=True)
    parser.add_argument('--seed', help='rng seed list', type=int, nargs='+', default=None, required=True)
    parser.add_argument('--obj', help='optim objective mode', type=str, default=None, required=True)
    parser.add_argument('--repo', help='repo dirpath list', type=str, nargs='+', default=None, required=True)
    parser.add_argument('--opt', help='opt mode: exact OR sampling', type=str, default=None, required=True)
    parser.add_argument('--custom', help='list of (key: value) items', type=str, nargs='+', default=[])
    arg = parser.parse_args()
    arg.cfg = arg.cfg.replace('file://','')
    return arg

if __name__ == '__main__':
    main()
