#!/bin/bash
NCPU=1
CFG=exsig.yaml

if [ "$#" -ne 0 ]; then
  echo "USAGE:"
  echo "bash run.sh"
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate nbwpgmain
REPODIR=~/ws/nbwpg

echo 'BEGIN --- '`date '+%Y-%m-%d %H:%M:%S'`
python $REPODIR/make_envprop_mesh.py \
--cfg $REPODIR/xprmt/envprop-mesh/$CFG \
--ncpu $NCPU
echo 'END --- '`date '+%Y-%m-%d %H:%M:%S'`
