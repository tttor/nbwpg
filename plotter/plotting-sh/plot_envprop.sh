#!/bin/bash
EXE=~/ws/nbwpg/plotter/envprop_mesh.py
CFG=cfg_envprop.yaml

$EXE \
--datadir ~/ws/nbwpg/data/envprop \
--cfg $CFG \
--data bs0 \
--mode 3d \
--show 1
