#!/bin/bash
NCPU=1
CFG=cfg/exsig.yaml

if [ "$#" -ne 0 ]; then
  echo "USAGE:"
  echo "bash run.sh"
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate nbwpgmain
REPODIR=~/ws/nbwpg

echo 'BEGIN --- '`date '+%Y-%m-%d %H:%M:%S'`
python $REPODIR/make_gradsamplingbasedexact_mesh.py \
--cfg $REPODIR/xprmt/biasgradexact-mesh/$CFG \
--ncpu $NCPU \
--repo $REPODIR
echo 'END --- '`date '+%Y-%m-%d %H:%M:%S'`
