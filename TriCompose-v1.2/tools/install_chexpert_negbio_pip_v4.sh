#!/usr/bin/env bash
# The CARC GCC module's inherited GCC_EXEC_PREFIX redirects old Conda GCC
# incorrectly. Select the dedicated environment's exact GCC 7.3.0 search root.
# No compiler/package/source version change. Prior failed scripts are immutable.
set -euo pipefail
workspace=/project2/ruishanl_1185/inference_3mod
export GCC_EXEC_PREFIX="$workspace/runtime/venvs/chexpert-negbio-py36-12682821-v1/lib/gcc/"
test -d "$GCC_EXEC_PREFIX/x86_64-conda_cos6-linux-gnu/7.3.0"
exec bash "$workspace/TriCompose-v1.2/tools/install_chexpert_negbio_pip_v3.sh"
