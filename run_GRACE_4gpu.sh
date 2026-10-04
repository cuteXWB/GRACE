#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"

SEED=666
GPUS=(0 1 2 3)
OUTPUT_ROOT="output_GRACE_all_seed666"
PRETRAINED_PATH="checkpoints/unimed_clip_vit_b16.pt"
RESUME="True"
EXTRA_OPTS=""
CLEAN_ROOT=0

usage() {
  cat <<'EOF'
Usage: bash run_GRACE_4gpu.sh [options]

  --seed <int>
  --gpus <0,1,2,3>
  --output-root <dir>
  --pretrained-path <path>
  --resume <True|False>
  --clean-root
  --extra-opts "<KEY VALUE ...>"

Phase 1 (four GPUs):
  GPU0 BUSI   -> BUSBRA,BUSUC,BUID,UDIAT
  GPU1 Kvasir -> ColonDB,ClinicDB,CVC300,BKAI
  GPU2 BTMRI  -> BRISC
  GPU3 ISIC   -> UWaterlooSkinCancer

Phase 2:
  GPU0 Covid19 ID only
  GPU1 EUS ID only (10 epochs from EUS_GRACE.yaml)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --gpus) IFS=',' read -r -a GPUS <<< "$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --pretrained-path) PRETRAINED_PATH="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --clean-root) CLEAN_ROOT=1; shift ;;
    --extra-opts) EXTRA_OPTS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done
export PYTHONHASHSEED="${SEED}"

if [[ ${#GPUS[@]} -lt 4 ]]; then
  echo "Four GPU IDs are required."
  exit 1
fi
if [[ ${CLEAN_ROOT} -eq 1 ]]; then
  rm -rf -- "${OUTPUT_ROOT}"
fi
mkdir -p "${OUTPUT_ROOT}/logs"

declare -a PIDS=()
declare -a NAMES=()

start_job() {
  local name="$1"
  local gpu="$2"
  local source="$3"
  local targets="$4"
  local log="${OUTPUT_ROOT}/logs/${name}.log"
  local command=(
    bash scripts/run_GRACE_id_ood.sh
    --seed "${SEED}"
    --gpu "${gpu}"
    --source-dataset "${source}"
    --percents 10,25,50,100
    --ood-targets "${targets}"
    --ood-source-percent 100
    --output-root "${OUTPUT_ROOT}/${source}"
    --pretrained-path "${PRETRAINED_PATH}"
    --resume "${RESUME}"
  )
  if [[ -n "${EXTRA_OPTS}" ]]; then
    command+=(--extra-opts "${EXTRA_OPTS}")
  fi
  echo "[start/${name}] GPU=${gpu} log=${log}"
  "${command[@]}" > "${log}" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${name}")
}

wait_jobs() {
  local failed=0
  for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
      echo "[done/${NAMES[$index]}]"
    else
      echo "[fail/${NAMES[$index]}]"
      failed=1
    fi
  done
  PIDS=()
  NAMES=()
  return "${failed}"
}

start_job BUSI "${GPUS[0]}" BUSI "BUSBRA,BUSUC,BUID,UDIAT"
start_job Kvasir "${GPUS[1]}" Kvasir "ColonDB,ClinicDB,CVC300,BKAI"
start_job BTMRI "${GPUS[2]}" BTMRI "BRISC"
start_job ISIC "${GPUS[3]}" ISIC "UWaterlooSkinCancer"
wait_jobs

start_job Covid19 "${GPUS[0]}" Covid19 ""
start_job EUS "${GPUS[1]}" EUS ""
wait_jobs

python scripts/summarize_all_results.py --root "${OUTPUT_ROOT}"
echo "All GRACE experiments completed under ${OUTPUT_ROOT}"
