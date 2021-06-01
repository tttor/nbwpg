#!/bin/bash
REPODIR=~/ws/nbwpg/plotter

$REPODIR/opt_mesh.py \
--gymdir ~/ws/nbwpg/gym-env \
--cfg cfg_optmesh_mat.yaml  \
--data bias_diff_abs \
--cbar 0 \
--cbarx 0
