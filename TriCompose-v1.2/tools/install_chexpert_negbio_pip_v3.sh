#!/usr/bin/env bash
# Old Conda GCC cannot discover cc1plus after prefix relocation; use its exact
# existing 7.3.0 executable, not another compiler or altered parser source.
set -euo pipefail
workspace=/project2/ruishanl_1185/inference_3mod
export COMPILER_PATH="$workspace/runtime/venvs/chexpert-negbio-py36-12682821-v1/libexec/gcc/x86_64-conda_cos6-linux-gnu/7.3.0"
test -x "$COMPILER_PATH/cc1plus"
exec bash "$workspace/TriCompose-v1.2/tools/install_chexpert_negbio_pip_v2.sh"
