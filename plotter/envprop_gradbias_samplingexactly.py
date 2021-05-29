#!/usr/bin/env python3
# https://stackoverflow.com/questions/48213884/transparent-error-bars-without-affecting-markers
# https://matplotlib.org/stable/gallery/text_labels_and_annotations/text_alignment.html
import os, pickle, argparse, yaml
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
import util_bwopt_plotter as u
from collections import defaultdict

def main():
    arg = parse_arg(); logdir = os.path.dirname(arg.cfg)
    with open(os.path.join(arg.cfg), 'r') as f:
        cfg = yaml.load(f)
        print(cfg)

    print('loading...')
    with open(os.path.join(logdir, 'data', cfg['mesh_fname']), 'rb') as f:
        meshdata = pickle.load(f)
        print(meshdata.keys()); print(meshdata['cfg'])

        tmix_list = list(meshdata['tmix'].values())
        ratio_cum = 0.; tmixrat_list = []
        for i, tmix in enumerate(sorted(list(set(tmix_list)))):
            n = tmix_list.count(tmix); ratio = n/len(tmix_list); ratio_cum += ratio
            tmixrat_list.append((tmix, n, ratio, ratio_cum))
            print('{} tmix {}: {} {:.3f} {:.3f}'.format(i+1, tmix, n, ratio, ratio_cum))

        print('most common tmix list')
        tmixrat_list = sorted(tmixrat_list, key=lambda entry: entry[1], reverse=True)
        for i, tmixrat in enumerate(tmixrat_list):
            print(i+1, tmixrat)
        u.print_stat(np.array(tmix_list), tag='tmix')

    print('organize...')
    data = {}
    key_list = ['premix_angerr', 'prepostmix_angerr', 'premix_normerr', 'prepostmix_normerr',
        'postmix_angerr', 'postmix_normerr']
    for k, v in meshdata.items():
        if k not in key_list:
            continue
        data[k] = defaultdict(list)
        for kk, vv in v.items():
            data[k][meshdata['tmix'][kk]].append(vv) # kk key: tmix target

    print('plotting...')
    for i, tmixrat in enumerate(tmixrat_list[0:6]):
        tmix, n, ratio, ratio_cum = tmixrat
        print('plotting', i+1, tmix, n, ratio, ratio_cum)

        premix_angerr = np.array(data['premix_angerr'][tmix])
        postmix_angerr = np.array(data['postmix_angerr'][tmix])
        prepostmix_angerr = np.array(data['prepostmix_angerr'][tmix])
        premix_angerr_mean = premix_angerr.mean(axis=u.sample_dimth)
        premix_angerr_std = premix_angerr.std(axis=u.sample_dimth)
        postmix_angerr_mean = postmix_angerr.mean(axis=u.sample_dimth)
        postmix_angerr_std = postmix_angerr.std(axis=u.sample_dimth)
        prepostmix_angerr_mean = prepostmix_angerr.mean(axis=u.sample_dimth)
        prepostmix_angerr_std = prepostmix_angerr.std(axis=u.sample_dimth)

        premix_normerr = np.array(data['premix_normerr'][tmix])
        postmix_normerr = np.array(data['postmix_normerr'][tmix])
        prepostmix_normerr = np.array(data['prepostmix_normerr'][tmix])
        premix_normerr_mean = premix_normerr.mean(axis=u.sample_dimth)
        premix_normerr_std = premix_normerr.std(axis=u.sample_dimth)
        postmix_normerr_mean = postmix_normerr.mean(axis=u.sample_dimth)
        postmix_normerr_std = postmix_normerr.std(axis=u.sample_dimth)
        prepostmix_normerr_mean = prepostmix_normerr.mean(axis=u.sample_dimth)
        prepostmix_normerr_std = prepostmix_normerr.std(axis=u.sample_dimth)

        y = premix_angerr_mean.tolist() + [prepostmix_angerr_mean]
        y = [yi*(180./np.pi) for yi in y] # to degree
        yerr = premix_angerr_std.tolist() + [prepostmix_angerr_std]
        yerr = [yerr_i*(180./np.pi) for yerr_i in yerr]
        x = range(len(y))
        y2 = premix_normerr_mean.tolist() + [prepostmix_normerr_mean]
        yerr2 = premix_normerr_std.tolist() + [prepostmix_normerr_std]
        assert np.isfinite(y).all()

        fig, ax = plt.subplots(figsize=(12, 9)); ax2 = ax.twinx()
        markers, caps, bars = ax.errorbar(x, y, yerr=yerr, color='blue', marker='', linestyle='-',
            linewidth=3, ecolor='blue', elinewidth=10)
        markers2, caps2, bars2 = ax2.errorbar(x, y2, yerr=yerr2, color='red', marker='', linestyle='-',
            linewidth=3, ecolor='red', elinewidth=3, capsize=3, capthick=3)

        ax.plot(len(x) - 1, postmix_angerr_mean, color='blue', marker='X', markersize=15)
        ax2.plot(len(x) - 1, postmix_normerr_mean, color='red', marker='X', markersize=15)

        fontsize = 45
        info = 'n(policy)= {} ({:.3f})\nt(mixing)= {}'.format(n, ratio, tmix)
        left, width = .25, .67; bottom, height = .25, .70
        right = left + width; top = bottom + height
        ax.text(right, top, info,
            fontdict={'size': fontsize, 'family': 'serif', 'color':  'black', 'weight': 'normal'},
            horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        xlim = ax.get_xlim()
        ax.hlines(0., *xlim, linewidth=1, alpha=1.0, linestyle='--', color='blue')
        ax.set_xlim(xlim)
        xlim2 = ax2.get_xlim()
        ax2.hlines(0., *xlim2, linewidth=1, alpha=1.0, linestyle='--', color='red')
        ax2.set_xlim(xlim2)

        fontsize = 25
        if cfg['xlabel']:
            ax.set_xlabel('$t$-th timestep so far', fontsize=fontsize)
        if cfg['yleftlabel']:
            ax.set_ylabel('Angular error (deg)', color='blue', fontsize=fontsize)
        if cfg['yrightlabel']:
            ax2.set_ylabel('Norm error', color='red', fontsize=fontsize)
        if ('xticks' in cfg.keys()) and (tmix in cfg['xticks'].keys()):
            ax.set_xticks(cfg['xticks'][tmix])
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelcolor='blue', labelsize=fontsize)
        _ = [bar.set_alpha(0.25) for bar in bars]
        ax2.tick_params(axis='y', labelcolor='red', labelsize=fontsize)
        _ = [bar.set_alpha(0.25) for bar in bars2]
        _ = [cap.set_alpha(0.25) for cap in caps2]

        envid = u.get_shortenvid(meshdata['cfg']['envid'])
        polnet = meshdata['cfg']['polnet']['mode']
        fname = '__'.join(['gradsamplingexact', str(i+1), polnet, envid]) + '.png'
        plotdir = os.path.join(logdir, 'gradsamplingexact-plot'); os.makedirs(plotdir, exist_ok=True)
        plt.savefig(os.path.join(plotdir, fname), dpi=300, bbox_inches='tight')
        plt.close(fig)

def parse_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', help='config fpath', type=str, default=None, required=True)
    arg = parser.parse_args()
    arg.cfg = arg.cfg.replace('file://','')
    return arg

if __name__ == '__main__':
    main()
