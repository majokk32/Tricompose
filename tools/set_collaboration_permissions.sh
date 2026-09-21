#!/usr/bin/env bash
set -euo pipefail

readonly TRICOMPOSE_WORKSPACE="/project2/ruishanl_1185/inference_3mod"
readonly TRICOMPOSE_GROUP="ruishanl_1185"
readonly TRICOMPOSE_OWNER="yikeyang"
readonly TRICOMPOSE_PROTECTED="${TRICOMPOSE_WORKSPACE}/artifacts/protected"

if [[ "$(id -un)" != "${TRICOMPOSE_OWNER}" ]]; then
  echo "ERROR: run this script as ${TRICOMPOSE_OWNER}." >&2
  exit 2
fi

case " $(id -Gn) " in
  *" ${TRICOMPOSE_GROUP} "*) ;;
  *)
    echo "ERROR: ${TRICOMPOSE_GROUP} is not active in this shell." >&2
    echo "Run 'newgrp ${TRICOMPOSE_GROUP}' and then rerun this script." >&2
    exit 2
    ;;
esac

if [[ ! -d "${TRICOMPOSE_PROTECTED}" ]]; then
  echo "ERROR: protected directory not found: ${TRICOMPOSE_PROTECTED}" >&2
  exit 2
fi

cd "${TRICOMPOSE_WORKSPACE}"

# Protected artifacts: project-group access only. Do not follow symlinks.
find -P "${TRICOMPOSE_PROTECTED}" -xdev -type d \
  -exec chgrp "${TRICOMPOSE_GROUP}" {} + \
  -exec chmod 2770 {} +
find -P "${TRICOMPOSE_PROTECTED}" -xdev -type f \
  -exec chgrp "${TRICOMPOSE_GROUP}" {} + \
  -exec chmod 0660 {} +

# Collaborative source/configuration trees. Preserve public read/execute bits,
# while enabling project-group edits and group inheritance for new files.
readonly -a TRICOMPOSE_CODE_DIRS=(
  "TriCompose-v1.0"
  "src"
  "tools"
  "tests"
  "configs"
  "schemas"
  "docs"
  "slurm"
)

# Some Jupyter cache entries were created by the NFS service account. Skip
# entries not owned by the invoking project owner instead of failing the whole
# permission update.
find -P "${TRICOMPOSE_CODE_DIRS[@]}" -user "${TRICOMPOSE_OWNER}" -type d \
  -exec chgrp "${TRICOMPOSE_GROUP}" {} + \
  -exec chmod g+rws {} +
find -P "${TRICOMPOSE_CODE_DIRS[@]}" -user "${TRICOMPOSE_OWNER}" -type f \
  -exec chgrp "${TRICOMPOSE_GROUP}" {} + \
  -exec chmod g+rw {} +

readonly -a TRICOMPOSE_ROOT_FILES=(
  "AGENTS.md"
  "README.md"
  "pyproject.toml"
  ".gitignore"
)
chgrp "${TRICOMPOSE_GROUP}" "${TRICOMPOSE_ROOT_FILES[@]}"
chmod g+rw "${TRICOMPOSE_ROOT_FILES[@]}"

if find -P "${TRICOMPOSE_PROTECTED}" -xdev -type d \
  ! -perm 2770 -print -quit | read -r; then
  echo "ERROR: a protected directory failed the mode audit." >&2
  exit 3
fi

if find -P "${TRICOMPOSE_PROTECTED}" -xdev -type f \
  ! -perm 0660 -print -quit | read -r; then
  echo "ERROR: a protected file failed the mode audit." >&2
  exit 3
fi

echo "Collaboration permissions applied successfully."
echo "Protected directories: logical_group=${TRICOMPOSE_GROUP}, mode=2770"
echo "Protected files: logical_group=${TRICOMPOSE_GROUP}, mode=0660"
echo "NOTE: the CARC NFS metadata view may display the logical project group as nobody."
