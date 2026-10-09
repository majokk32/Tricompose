#!/usr/bin/env bash
# Exact upstream packages/old GCC, with verified explicit -B/sysroot paths.
# v1-v4 failed/cancelled setup attempts remain unchanged for auditability.
set -euo pipefail
umask 007
workspace=/project2/ruishanl_1185/inference_3mod
parser_env="$workspace/runtime/venvs/chexpert-negbio-py36-12682821-v1"
setup_tmp="$workspace/.tmp/chexpert_negbio_setup_12682821_001"
export TMPDIR="$setup_tmp"
export PYTHONDONTWRITEBYTECODE=1
export CONDARC="$workspace/TriCompose-v1.2/configs/chexpert_negbio_condarc_v1.yml"
export CONDA_REGISTER_ENVS=false
export CONDA_PKGS_DIRS="$workspace/.cache/chexpert_negbio/conda_pkgs"
export CONDA_ENVS_PATH="$workspace/runtime/venvs"
export XDG_CACHE_HOME="$workspace/.cache/chexpert_negbio/xdg_cache"
export XDG_CONFIG_HOME="$workspace/.cache/chexpert_negbio/xdg_config"
export XDG_DATA_HOME="$workspace/.cache/chexpert_negbio/xdg_data"
export PIP_CONFIG_FILE=/dev/null
export PIP_CACHE_DIR="$workspace/.cache/chexpert_negbio/pip"
export PIP_CERT=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export PIP_DISABLE_PIP_VERSION_CHECK=1
export MAKEFLAGS=-j4
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$workspace"
"$workspace/runtime/venvs/report-context-v12-12576792/bin/python" -c \
  'import sys; sys.path.insert(0,"TriCompose-v1.2/tools"); from score_cached_opacity_candidates import guard; guard()'
test -f "$parser_env/conda-meta/history"
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
set +u
conda activate "$parser_env"
set -u
parser_target=x86_64-conda_cos6-linux-gnu
parser_flags="-B$parser_env/libexec/gcc/$parser_target/7.3.0/ -B$parser_env/lib/gcc/$parser_target/7.3.0/ -B$parser_env/$parser_target/bin/ -B$parser_env/$parser_target/sysroot/usr/lib/ --sysroot=$parser_env/$parser_target/sysroot"
test -f "$parser_env/lib/gcc/$parser_target/7.3.0/crtbeginS.o"
test -f "$parser_env/$parser_target/sysroot/usr/lib/crti.o"
export CC="$parser_env/bin/$parser_target-gcc $parser_flags"
export CXX="$parser_env/bin/$parser_target-g++ $parser_flags"
export LDSHARED="$CC -shared"
python -m pip install --requirement "$workspace/TriCompose-v1.2/configs/chexpert_negbio_pip_exact_v1.txt"
