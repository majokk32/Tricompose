#!/usr/bin/env bash
# User-terminal download only. No credential files, clinical inspection or models.
set +x
set -euo pipefail

readonly TC_WORKSPACE=/project2/ruishanl_1185/inference_3mod
readonly TC_PARENT="$TC_WORKSPACE/artifacts/protected/tricompose_v1_2/reference_datasets"
readonly TC_SOURCE=https://physionet.org/files/cxrgraph/1.0.0/
readonly TC_ACCEPT='^https://physionet[.]org/files/cxrgraph/1[.]0[.]0/($|manual_data/|SHA256SUMS[.]txt$|LICENSE[.]txt$)'

case "${1:-}" in
  '') TC_DRY_RUN=0 ;;
  --dry-run) TC_DRY_RUN=1 ;;
  --help)
    printf '%s\n' 'Run interactively: bash tools/download_cxrgraph_manual.sh' \
      'Enter your PhysioNet password only at the hidden wget prompt.' \
      'Downloads manual annotations and checksums into a fresh protected directory.' \
      'Does not download the bulk automatic annotations or run inference.'
    exit 0 ;;
  *) printf '%s\n' 'Unknown option; use --help.' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { printf '%s\n' 'Unexpected arguments.' >&2; exit 2; }

[[ -d "$TC_PARENT" && ! -L "$TC_PARENT" && "$(readlink -f "$TC_PARENT")" == "$TC_PARENT" ]] || {
  printf '%s\n' 'Expected non-symlink protected parent is unavailable.' >&2; exit 2;
}
[[ "$(stat -c %a "$TC_PARENT")" == 2770 ]] || {
  printf '%s\n' 'Protected parent must have mode 2770.' >&2; exit 2;
}
case "$(stat -c %g "$TC_PARENT")" in
  96293|65534) ;;
  *) printf '%s\n' 'Protected parent must use the CARC project group.' >&2; exit 2 ;;
esac
command -v wget >/dev/null || { printf '%s\n' 'GNU wget is required.' >&2; exit 2; }

tc_command() {
  TC_WGET=(wget --no-config --no-netrc --no-hsts --no-cookies
    -r -N -c -np --level=2 --https-only --max-redirect=0
    --execute=robots=off --domains=physionet.org --accept-regex="$TC_ACCEPT"
    --timeout=30 --tries=2 --quota=100m --user=majokk --ask-password
    --no-host-directories --cut-dirs=3 --directory-prefix="$TC_DOWNLOAD_DIR"
    --output-file="$TC_DOWNLOAD_DIR/private_logs/wget.log" "$TC_SOURCE")
}

if [[ "$TC_DRY_RUN" == 1 ]]; then
  TC_DOWNLOAD_DIR='<fresh protected directory>'
  tc_command
  printf '%s\n' 'status: dry_run_no_download_no_credentials_no_directory_creation'
  printf 'command: '; printf '%q ' "${TC_WGET[@]}"; printf '\n'
  exit 0
fi

[[ -t 0 && -t 1 ]] || {
  printf '%s\n' 'Run this script in your own interactive terminal; do not send a password to chat.' >&2
  exit 2
}
# Do not invoke inherited credential helpers or read .wgetrc/.netrc/cookies/HSTS.
unset WGET_ASKPASS SSH_ASKPASS
umask 0007
TC_DOWNLOAD_DIR=$(mktemp -d "$TC_PARENT/cxrgraph_manual_XXXXXXXX")
chmod 2770 "$TC_DOWNLOAD_DIR"
mkdir -m 2770 "$TC_DOWNLOAD_DIR/private_logs"

tc_permissions() {
  if [[ -n "$(find "$TC_DOWNLOAD_DIR" -type l -print -quit)" ]]; then
    printf '%s\n' 'Unexpected symlink in download; stop without following it.' >&2
    return 2
  fi
  find "$TC_DOWNLOAD_DIR" -type d -exec chmod 2770 {} +
  find "$TC_DOWNLOAD_DIR" -type f -exec chmod 0660 {} +
  [[ "$(find "$TC_DOWNLOAD_DIR" ! -gid 96293 ! -gid 65534 -printf . | wc -c)" == 0 ]] || {
    printf '%s\n' 'Unexpected ownership; CARC project-group permissions need checking.' >&2
    return 2
  }
}
trap tc_permissions EXIT

printf 'Download directory: %s\n' "$TC_DOWNLOAD_DIR"
printf '%s\n' 'Enter the PhysioNet account password at the hidden wget prompt; not a Hugging Face token.'
tc_command
if "${TC_WGET[@]}"; then
  tc_permissions
else
  TC_WGET_STATUS=$?
  printf 'Download failed (wget exit %s). Private log: %s/private_logs/wget.log\n' "$TC_WGET_STATUS" "$TC_DOWNLOAD_DIR" >&2
  printf '%s\n' 'Partial downloads are retained. No clinical file was inspected.' >&2
  exit "$TC_WGET_STATUS"
fi

readonly TC_TEST="$TC_DOWNLOAD_DIR/manual_data/test.json"
readonly TC_SUMS="$TC_DOWNLOAD_DIR/SHA256SUMS.txt"
[[ -s "$TC_TEST" && -s "$TC_SUMS" ]] || {
  printf '%s\n' 'Required manual test file or official checksum list is missing; not benchmark-ready.' >&2
  exit 2
}
# Parse only the checksum list, never the clinical JSON. Do not follow arbitrary
# filenames from a remote checksum list with sha256sum --check.
TC_EXPECTED=$(awk '$2 == "manual_data/test.json" || $2 == "./manual_data/test.json" || $2 == "*manual_data/test.json" {print $1}' "$TC_SUMS")
[[ "$TC_EXPECTED" =~ ^[0-9a-f]{64}$ ]] || {
  printf '%s\n' 'A unique official checksum for manual_data/test.json is required.' >&2; exit 2;
}
TC_ACTUAL=$(sha256sum "$TC_TEST" | awk '{print $1}')
[[ "$TC_ACTUAL" == "$TC_EXPECTED" ]] || {
  printf '%s\n' 'Manual test checksum mismatch; do not start evaluation.' >&2; exit 2;
}
printf '%s\n' 'status: manual_test_downloaded_checksum_verified'
printf 'manual test path: %s\n' "$TC_TEST"
printf 'manual test sha256: %s\n' "$TC_ACTUAL"
printf '%s\n' 'No model inference or clinical schema evaluation has run. Share the path/status only.'
