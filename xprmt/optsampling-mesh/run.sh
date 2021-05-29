#!/bin/bash
NCPU=1
CFG=cfg/exsig.yaml

# OBJ=bias
# COND=fisherbias
# COND=identity
OBJ=gainbiasbarrier_100.0
COND=fisherbiasfisherbarrier

SEEDS='1234567 7654321'

if [ "$#" -ne 0 ]; then
  echo "USAGE:"
  echo "bash run.sh"
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate nbwpgmain
REPODIR=~/ws/nbwpg

echo 'BEGIN --- '`date '+%Y-%m-%d %H:%M:%S'`
python $REPODIR/main/make_opt_mesh.py \
--opt sampling \
--cfg $REPODIR/xprmt/optsampling-mesh/cfg/$CFG \
--seed $SEEDS \
--ncpu $NCPU \
--obj $OBJ \
--custom conditioner_mode $COND \
--repo $REPODIR
echo 'END --- '`date '+%Y-%m-%d %H:%M:%S'`
