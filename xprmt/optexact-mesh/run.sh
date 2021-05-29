#!/bin/bash
NCPU=1
CFG=cfg/exsig.yaml

OBJ=gain
COND=fisher_steady
# OBJ=bias
# COND=identity
# COND=hess
# COND=fisher_transient_withsteadymul_upto_t1
# OBJ=disc_0.99
# COND=fisher_disc
# OBJ=gainbiasbarrier_100.0_1.0
# COND=fisher

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
--opt exact \
--cfg $REPODIR/xprmt/optexact-mesh/$CFG \
--seed 1234567 \
--ncpu $NCPU \
--obj $OBJ \
--custom conditioner_mode $COND \
--repo $REPODIR
echo 'END --- '`date '+%Y-%m-%d %H:%M:%S'`
