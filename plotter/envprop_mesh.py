#!/usr/bin/env python3
import os, pickle, numpy as np, argparse, yaml
import matplotlib.pyplot as plt
import util_bwopt_plotter as u
from matplotlib import cm
from matplotlib.colors import LightSource, ListedColormap
from mpl_toolkits.mplot3d import Axes3D # This import registers the 3D projection, but is otherwise unused.

def main():
    arg = parse_arg(); logdir = os.path.dirname(arg.cfg)
    with open(arg.cfg, 'r') as f:
        cfg = yaml.load(f); cfg['logdir'] = logdir
        envcfg_dpath = os.path.join(arg.gymsymboldir, 'gym_symbol/envs/config')
        envcfg_fname = cfg['envid'].replace('-', '_').lower() + '.yaml'
        with open(os.path.join(envcfg_dpath, envcfg_fname), 'r') as f2:
            envcfg = yaml.load(f2)
            cfg = {**cfg, **envcfg['deterministic_policy']}

    print('loading...')
    datadir = os.path.join(logdir, 'data', cfg['mesh_dname'])
    meshdata = {}
    for key in ['wxmat', 'wymat'] +  arg.data:
        with open(os.path.join(datadir, key+'.pkl'), 'rb') as f:
            meshdata = {**meshdata, **pickle.load(f)}

    meshmeta = meshdata['cfg']; assert cfg['envid'] in meshmeta['envid']
    xs, ys = meshdata['wxmat'], meshdata['wymat'] # the parameter(=[weight, bias]) mesh
    zs_list = []
    for dkey in arg.data:
        zs = np.round(meshdata[dkey], meshmeta['round_decimal'])
        print(dkey, zs.shape)
        if zs.shape[2]==1:
            u.print_stat(zs, tag=dkey)
        zs_list.append(zs)

    plot_fn = {'con': plot_contour, '3d': plot_3d, 'mat': plot_mat}
    plot_fn[arg.mode](xs, ys, zs_list, arg, cfg, meshmeta)

def plot_3d(xs, ys, zs, arg, cfg, meta):
    # https://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
    # https://docs.enthought.com/mayavi/mayavi/auto/mlab_camera.html
    # https://docs.enthought.com/mayavi/mayavi/auto/mlab_figure.html
    # https://docs.enthought.com/mayavi/mayavi/tips.html
    # https://docs.enthought.com/mayavi/mayavi/auto/example_wigner.html
    # https://scipy-lectures.org/packages/3d_plotting/index.html
    # https://stackoverflow.com/questions/23701148/setting-the-color-of-mayavi-mlab-mesh-axes-labels
    assert cfg['resolution_step']==1
    from mayavi import mlab
    zmode = arg.data[0]; zs = zs[0]; print('plot 3d', zmode)
    zs = zs.squeeze(axis=-1)
    fstamp = '__'.join([zmode, cfg['envid'], meta['polnet']['mode']])

    # cbmin = cbmax = None
    cbmin = cfg[zmode + '_min']; cbmax = cfg[zmode + '_max'] # of deterministic policies
    extent = [np.min(xs), np.max(xs), np.min(ys), np.max(ys), np.min(zs), np.max(zs)]
    zs_masked = np.ma.masked_invalid(zs)
    print('cbmin', cbmin, 'cbmax', cbmax)
    assert (np.min(zs) >= cbmin) and (np.max(zs) <= cbmax)

    if 'ds0_' in zmode:
        gamma = float(zmode.replace('ds0_', ''))
        zs = (1 - gamma)*zs # scale disrew value
        cbmin = (1 - gamma)*cbmin; cbmax = (1 - gamma)*cbmax
        u.print_stat(zs, tag=zmode+'(scaled)')
        print('cbmin(scaled)', cbmin, 'cbmax(scaled)', cbmax)

    mlab.options.offscreen = not(arg.show)
    fig = mlab.figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.), figure=fstamp, size=(800, 700))

    su = mlab.surf(xs, ys, zs, colormap='plasma', vmin=cbmin, vmax=cbmax,
                   extent=extent, warp_scale=1.)
    mlab.outline(su)
    mlab.axes(su, ranges=extent, nb_labels=5, xlabel='', ylabel='', zlabel='')

    # Mesh is for plotting data with Nan (Surf should work, but it does not so far)
    # me = mlab.mesh(xs, ys, zs, colormap='plasma', mask=zs_masked.mask)
    # mlab.outline(me)
    # mlab.axes(me, ranges=extent)

    # mlab.view(azimuth=0, elevation=0, distance=cfg[zmode]['myv_view_distance'])
    try:
        dist = cfg[zmode]['myv_view_distance']
    except:
        dist = None
    mlab.view(azimuth=0, elevation=0, distance=dist)
    mlab.orientation_axes()

    if arg.show:
        mlab.show() # should set `mlab.options.offscreen = False` above
    else:
        plotdir = os.path.join(cfg['logdir'], 'plot-3d'); os.makedirs(plotdir, exist_ok=True)
        fpath = os.path.join(plotdir, fstamp + '.png')
        mlab.savefig(fpath)

def plot_contour(xs, ys, zs, arg, cfg, meta):
    zmode = arg.data
    print('plot contour', zmode)
    xlabel = 'x: weight'; ylabel = 'y: bias'
    levels = cfg['con']['levels']

    fval = zs[0].squeeze(axis=-1)
    max0 = np.max(fval)
    max0_idx_i, max0_idx_j = np.where(fval==max0)
    max0_x = xs[max0_idx_i, max0_idx_j]
    max0_y = ys[max0_idx_i, max0_idx_j]
    print('max0', max0, 'len(max0_idx_i)', len(max0_idx_i))
    print('max0_x, max0_y', max0_x, max0_y)

    print('plotting...')
    fig, ax = plt.subplots(figsize=(12, 9))
    cx = ax.contour(xs, ys, fval, levels, cmap=cm.Blues, alpha=1.0)
    ax.clabel(cx, inline=1, fontsize=10)
    ax.plot(max0_x, max0_y, color='red', marker='*', markersize=10)
    print('cx.levels', ', '.join([str(np.round(lvl, 3)) for lvl in cx.levels]))

    if len(zmode)>=3: # overlay grad-like field
        if 'hvp' in zmode[1]:
            assert ('hvp' in zmode[2])
            zs[1] = - zs[1]; zs[2] = - zs[2] # make positive definite for ascent

        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.quiver.html
        # To plot vectors in the x-y plane, with u and v having the same units as x and y,
        # use angles='xy', scale_units='xy', scale=1
        # 'xy': Arrows point from (x,y) to (x+u, y+v).
        # Use this for plotting a gradient field, for example.
        # assert arg.ss is not None
        stepsize = arg.ss; xlabel += ' (stepsize {})'.format(stepsize)

        # downsampling
        step = 4
        xs_quiver = xs[::step, ::step]
        ys_quiver = ys[::step, ::step]
        zxs_quiver = zs[1][::step, ::step]
        zys_quiver = zs[2][::step, ::step]

        # plot bias only on gain-optimal region, satisfying 2nd order stationary condition
        if 'bias' in zmode[0]:
            assert 'bias' in zmode[1] and 'bias' in zmode[2]
            assert len(zs)==len(zmode)==4 # 3: gaingrad_norm
            assert arg.eps is not None
            gaingrad_norm = zs[3][::step, ::step]
            xs_quiver = xs_quiver[(gaingrad_norm <= arg.eps)]
            ys_quiver = ys_quiver[(gaingrad_norm <= arg.eps)]
            zxs_quiver = zxs_quiver[(gaingrad_norm <= arg.eps)]
            zys_quiver = zys_quiver[(gaingrad_norm <= arg.eps)]
            xlabel += ' (eps {})'.format(arg.eps)

        scale = 1/stepsize if arg.ss is not None else None
        ax.quiver(xs_quiver, ys_quiver, zxs_quiver, zys_quiver,
            units='xy', angles='xy', scale_units='xy', scale=scale, color='magenta')

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(axis='both'); ax.set_aspect('equal')
    if 'xlim' in cfg.keys():
        xmin, xmax = cfg['xlim']
        ax.set_xlim(xmin=xmin, xmax=xmax)
    if 'ylim' in cfg.keys():
        ymin, ymax = cfg['ylim']
        ax.set_ylim(ymin=ymin, ymax=ymax)

    print('saving...')
    fstamp = '__'.join([zmode[0], cfg['envid'], meta['polnet']['mode']])
    plotdir = os.path.join(cfg['logdir'], 'plot-con'); os.makedirs(plotdir, exist_ok=True)
    plt.savefig(os.path.join(plotdir, fstamp + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_mat(xs, ys, zs_list, arg, cfg, meta):
    zmode = arg.data[0]
    step = cfg['resolution_step']
    xs = xs[::step, ::step]
    ys = ys[::step, ::step]
    zs = zs_list[0][::step, ::step]

    if 'grad_' in zmode:
        print(zs.shape)
        eps = float(cfg['grad_g']['eps'])
        zs = np.linalg.norm(zs, ord=None, axis=2)
        zs[zs > eps] = np.nan
        u.print_stat(zs)

        assert arg.data[1]=='g'
        zs1 = zs_list[1][::step, ::step]
        zs1[zs > eps] = np.nan
        zs1[zs1 < 3.27385953] = np.nan
        u.print_stat(zs1, tag=arg.data[1])
        zs1, x_sorted, y_sorted = u.arrange_mesh2mat(xs, ys, zs1, ndecimal=8)
        exit()

    zs, x_sorted, y_sorted = u.arrange_mesh2mat(xs, ys, zs, ndecimal=8)
    if 'xticks' in cfg.keys():
        assert 'yticks' in cfg.keys()
        xticktargets = cfg['xticks']
        xticks = [x_sorted.index(i) for i in xticktargets]
        xticklabels = ['{:.1f}'.format(i) for i in xticktargets]
        yticktargets = cfg['yticks']
        yticks = [y_sorted.index(i) for i in yticktargets]
        yticklabels = ['{:.1f}'.format(i) for i in yticktargets]

    cmap = 'plasma'
    cmap_ticks = None

    print('plot mat', zmode)
    fig, ax = plt.subplots(figsize=(12, 9))
    cax = ax.matshow(zs, origin='lower', cmap=cmap, interpolation='none')
    if 'xticks' in cfg.keys():
        assert 'yticks' in cfg.keys()
        xticktargets = cfg['xticks']
        xticks = [x_sorted.index(i) for i in xticktargets]
        xticklabels = ['{:.1f}'.format(i) for i in xticktargets]
        yticktargets = cfg['yticks']
        yticks = [y_sorted.index(i) for i in yticktargets]
        yticklabels = ['{:.1f}'.format(i) for i in yticktargets]
        ax.set_xticks(xticks); ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks); ax.set_yticklabels(yticklabels)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.grid(axis='both'); ax.set_aspect('auto')
    fig.colorbar(cax, ticks=cmap_ticks, label='')

    print('saving...')
    fstamp = '__'.join([zmode, cfg['envid'], meta['polnet']['mode']])
    plotdir = os.path.join(cfg['logdir'], 'plot-mat'); os.makedirs(plotdir, exist_ok=True)
    plt.savefig(os.path.join(plotdir, fstamp + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # print('plot mat', arg.data[1])
    # fig, ax = plt.subplots(figsize=(12, 9))
    # cax = ax.matshow(zs1, origin='lower', cmap=cmap, interpolation='none')
    # if cfg['xticks'] is not None:
    #     ax.set_xticks(xticks); ax.set_xticklabels(xticklabels)
    #     ax.set_yticks(yticks); ax.set_yticklabels(yticklabels)
    # ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    # ax.grid(axis='both'); ax.set_aspect('auto')
    # fig.colorbar(cax, ticks=cmap_ticks, label='')

    # print('saving...')
    # fstamp = '__'.join([arg.data[1], cfg['envid'], meta['polnet']['mode']])
    # plotdir = os.path.join(cfg['logdir'], 'plot-mat'); os.makedirs(plotdir, exist_ok=True)
    # plt.savefig(os.path.join(plotdir, fstamp + '.png'), dpi=300, bbox_inches='tight')
    # plt.close(fig)

def parse_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', help='config filepath', type=str, default=None, required=True)
    parser.add_argument('--data', help='data type', type=str, nargs='+', default=None, required=True)
    parser.add_argument('--mode', help='plot mode', type=str, default=None, required=True)
    parser.add_argument('--gymsymboldir', help='gym-symbol dir, eg `~/gym-symbol`', type=str, required=True)
    parser.add_argument('--show', help='show plot', type=int, default=0)

    arg = parser.parse_args()
    arg.cfg = arg.cfg.replace('file://','')
    return arg

if __name__ == '__main__':
    main()
