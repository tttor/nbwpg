#!/usr/bin/env python3
# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/multi_image.html#sphx-glr-gallery-images-contours-and-fields-multi-image-py
# https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.axes.Axes.imshow.html#matplotlib.axes.Axes.imshow
# https://stackoverflow.com/questions/5263034/remove-colorbar-from-figure-in-matplotlib
# https://matplotlib.org/3.1.1/gallery/pyplots/text_layout.html#sphx-glr-gallery-pyplots-text-layout-py
# https://stackoverflow.com/questions/52767798/reduce-horizontal-colorbar-padding
import os, pickle, argparse, yaml
import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
import util_bwopt_plotter as u

def main():
    arg = parse_arg()
    with open(os.path.join(arg.cfg), 'r') as f:
        cfg = yaml.load(f)
        envcfg_dpath = os.path.join(arg.gymdir, 'gym-symbol/gym_symbol/envs/config')
        envcfg_fname = cfg['envid'].replace('-', '_').lower() + '.yaml'
        with open(os.path.join(envcfg_dpath, envcfg_fname), 'r') as f2:
            envcfg = yaml.load(f2)
            cfg = {**cfg, **envcfg['deterministic_policy']}
        print(cfg)

    for spec_idx, spec in enumerate(cfg['spec_list']):
        print('plotting spec_idx', spec_idx, len(cfg['spec_list'])-1, '=======')
        plot(spec, cfg, arg)

def plot(spec, cfg, arg):
    print('loading...')
    def check_spec(cfg):
        for k, v in spec.items():
            if cfg[k] != v:
                return False
        return True

    logdir = os.path.dirname(arg.cfg); datadir = os.path.join(logdir, 'data')
    zs_list = []; seed_list = []
    for mesh_fname in os.listdir(datadir):
        if '.pkl' not in mesh_fname:
            continue
        with open(os.path.join(datadir, mesh_fname), 'rb') as f:
            meshdata = pickle.load(f); seed = meshdata['seed']
            # print(meshdata.keys()); print(meshdata['cfg']); exit()
            if not(check_spec(meshdata['cfg'])):
                continue
            if ('commonseed_list' in cfg.keys()) and (seed not in cfg['commonseed_list']):
                continue
            print(mesh_fname[0:200], seed)
            seed_list.append(seed)
            xs, ys = meshdata['wxmat'], meshdata['wymat'] # the parameter(=[weight, bias]) mesh
            obj = arg.data.split('_')[0]
            zs = meshdata[obj]
            zs, x_sorted, y_sorted = u.arrange_mesh2mat(xs, ys, zs, meshdata['cfg']['round_decimal'])
            if obj == 'disc':
                gamma = meshdata['discountfactor'][0, 0] # same for all mesh coordinates
                zs = (1 - gamma)*zs
            if '_diff' in arg.data:
                if obj=='disc':
                    gamma_str = '{:.2f}'.format(gamma) if (gamma <= 0.99) else '{:.8f}'.format(gamma)
                    max_value = (1 - gamma)*cfg['ds0_'+gamma_str+'_max']
                elif obj=='bias':
                    max_value = cfg['bs0max_gainmax']
                else:
                    raise NotImplementedError(obj)
                zs = max_value - zs
            if '_ratio' in arg.data:
                zs = zs/max_value
            if '_abs' in arg.data:
                zs = np.abs(zs)
            zs_list.append(zs)
    seed_list = sorted(seed_list)
    print(len(seed_list), seed_list, 'commonseed=', 'commonseed_list' in cfg.keys())
    assert len(seed_list) > 0; assert len(set(seed_list))>=len(seed_list)

    zs = np.nanmean(np.stack(zs_list), axis=u.sample_dimth)
    u.print_stat(zs, tag=arg.data); # print('x_sorted', x_sorted); print('y_sorted', y_sorted)

    print('plotting mat', arg.data)
    fig, ax = plt.subplots(figsize=(12, 9))
    label_fontsize = 25; tick_fontsize = 20
    text_halign = 'center'; text_valign = 'center'
    try:
        text_size = cfg['text_size'][arg.data]
    except:
        text_size = 60
    if arg.data.replace('_abs', '').replace('_ratio', '') in ['bias_diff', 'gain_diff', 'disc_diff']:
        cmap = mpl.cm.cool
    elif arg.data in ['fval', 'bias', 'gain', 'disc', 'gainopt_fval']:
        cmap = mpl.cm.plasma
    elif arg.data in ['gainopt_converged', 'rollback']:
        cmap = mpl.cm.rainbow
    else:
        raise NotImplementedError(arg.data)
    assert arg.data in cfg['colorbar_ticks'] # decided to strictly enforce this
    if arg.data in cfg['colorbar_ticks']:
        cb_ticks = cfg['colorbar_ticks'][arg.data]
        cb_tickmin = min(cfg['colorbar_ticks'][arg.data])
        cb_tickmax = max(cfg['colorbar_ticks'][arg.data])
        print("cfg['colorbar_ticks']['force']", cfg['colorbar_ticks']['force'])
        if cfg['colorbar_ticks']['force']:
            print('WARN: affect the statistic that will be taken later!')
            zs[zs < cb_tickmin] = cb_tickmin; zs[zs > cb_tickmax] = cb_tickmax
        else:
            assert (zs[~np.isnan(zs)] <= cb_tickmax).all()
            assert (zs[~np.isnan(zs)] >= cb_tickmin).all()
    else:
        cb_tickmin = min(zs[~np.isnan(zs)])
        cb_tickmax = max(zs[~np.isnan(zs)])
        cb_ticks = [cb_tickmin, cb_tickmax]
    print('cb_tickmin {} cb_tickmax {}'.format(cb_tickmin, cb_tickmax))

    norm = mpl.colors.Normalize(vmin=cb_tickmin, vmax=cb_tickmax)
    cax = ax.matshow(zs, origin='lower', cmap=cmap, interpolation='none', norm=norm)
    cb = fig.colorbar(cax, ticks=cb_ticks, label='', pad=0.01)
    if arg.cbar==0:
        cb.remove()

    info = '${:.3f} \pm {:.3f}$'.format(np.nanmean(zs), np.nanstd(zs))
    ax.text(0.5, 0.5, info,
        fontdict={'size': text_size, 'family': 'serif', 'color':  'black', 'weight': 'normal'},
        horizontalalignment=text_halign, verticalalignment=text_valign, transform=ax.transAxes)

    # ax.set_xlabel('1st weight $(x)$', fontsize=label_fontsize)
    # ax.set_ylabel('2nd weight $(y)$', fontsize=label_fontsize)
    if ('xticks' in cfg.keys()) and ('yticks' in cfg.keys()):
        xticktargets = cfg['xticks']
        xticks = [x_sorted.index(i) for i in xticktargets]
        xticklabels = ['{:.1f}'.format(i) for i in xticktargets]
        yticktargets = cfg['yticks']
        yticks = [y_sorted.index(i) for i in yticktargets]
        yticklabels = ['{:.1f}'.format(i) for i in yticktargets]
        ax.set_xticks(xticks); ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks); ax.set_yticklabels(yticklabels)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False, labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(axis='both'); ax.set_aspect('auto')

    print('writing')
    envid = u.get_shortenvid(meshdata['cfg']['envid'])
    polnet = meshdata['cfg']['polnet']['mode']
    tag = [arg.data] + ['{}={}'.format(k, v) for k, v in spec.items()]
    tag += ['nseed{}'.format(len(zs_list)), polnet, envid]
    fname = '__'.join(tag)
    plt.savefig(os.path.join(logdir, fname + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    if arg.cbarx==1:
        print('writing colorbar standalone ...')
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_axes([0.05, 0.05, 0.05, 0.9])
        cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical',
            ticks=cfg['colorbar_ticks'][arg.data])
        # cb.set_label(label='Bias $v_b(s_0, \mathbf{\\theta})$ at the final iteration', fontsize=label_fontsize)
        for yt in cb.ax.get_yticklabels():
            yt.set_fontsize(tick_fontsize)

        fname = '__'.join([fname, 'colorbar'])
        plt.savefig(os.path.join(logdir, fname + '.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

def parse_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', help='config fpath', type=str, default=None, required=True)
    parser.add_argument('--data', help='data type', type=str, default=None, required=True)
    parser.add_argument('--gymdir', help='customized gym-env dir', type=str, required=True)
    parser.add_argument('--cbar', help='colorbar mode', type=int, default=1)
    parser.add_argument('--cbarx', help='colorbar: standalone', type=int, default=0)
    arg = parser.parse_args()
    arg.cfg = arg.cfg.replace('file://','')
    return arg

if __name__ == '__main__':
    main()
