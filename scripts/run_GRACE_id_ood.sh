#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

SEED=666
GPU=""
OUTPUT_ROOT="output_GRACE_id_ood_seed666"
SOURCE_DATASET="BUSI"
PERCENTS=(10 25 50 100)
OOD_SOURCE_PERCENT=100
OOD_TARGETS=("BUSBRA" "BUSUC" "BUID" "UDIAT")
PRETRAINED_PATH="checkpoints/unimed_clip_vit_b16.pt"
PROMPT_DESIGN="original"
RESUME="True"
CLEAN_ROOT=0
EXTRA_OPTS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/run_GRACE_id_ood.sh [options]

  --seed <int>
  --gpu <id>
  --output-root <dir>
  --source-dataset <name>
  --percents <10,25,50,100>
  --ood-targets <a,b,c>          Empty string runs ID only
  --ood-source-percent <int>
  --pretrained-path <path>
  --prompt-design <name>
  --resume <True|False>
  --clean-root
  --extra-opts "<KEY VALUE ...>"

Paper-aligned convenience overrides:
  --use-gvos <True|False>
  --use-gcfp <True|False>
  --use-pgc <True|False>
  --use-meo <True|False>
  --lambda-virt <float>
  --lambda-clip <float>
  --lambda-cons <float>
  --k <int>
  --m <int>
EOF
}

append_opt() {
  EXTRA_OPTS+=("$1" "$2")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --source-dataset) SOURCE_DATASET="$2"; shift 2 ;;
    --percents) IFS=',' read -r -a PERCENTS <<< "$2"; shift 2 ;;
    --ood-targets)
      if [[ -z "$2" ]]; then OOD_TARGETS=(); else IFS=',' read -r -a OOD_TARGETS <<< "$2"; fi
      shift 2
      ;;
    --ood-source-percent) OOD_SOURCE_PERCENT="$2"; shift 2 ;;
    --pretrained-path) PRETRAINED_PATH="$2"; shift 2 ;;
    --prompt-design) PROMPT_DESIGN="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --clean-root) CLEAN_ROOT=1; shift ;;
    --extra-opts) read -r -a USER_OPTS <<< "$2"; EXTRA_OPTS+=("${USER_OPTS[@]}"); shift 2 ;;
    --use-gvos) append_opt MODEL.USE_GVOS "$2"; shift 2 ;;
    --use-gcfp) append_opt MODEL.USE_GCFP "$2"; shift 2 ;;
    --use-pgc) append_opt MODEL.USE_PGC "$2"; shift 2 ;;
    --use-meo) append_opt MODEL.PGC_USE_MEO "$2"; shift 2 ;;
    --lambda-virt) append_opt MODEL.LAMBDA_VIRT "$2"; shift 2 ;;
    --lambda-clip) append_opt MODEL.LAMBDA_CLIP "$2"; shift 2 ;;
    --lambda-cons) append_opt MODEL.LAMBDA_CONS "$2"; shift 2 ;;
    --k) append_opt MODEL.GVOS_STEPS "$2"; shift 2 ;;
    --m) append_opt MODEL.GCFP_NUM_PROBES "$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done
export PYTHONHASHSEED="${SEED}"

if [[ -n "${GPU}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

SOURCE_CONFIG="configs/${SOURCE_DATASET}_GRACE.yaml"
if [[ ! -f "${SOURCE_CONFIG}" ]]; then
  echo "Source config not found: ${SOURCE_CONFIG}"
  exit 1
fi
if [[ ! -f "${PRETRAINED_PATH}" ]]; then
  echo "UniMedCLIP checkpoint not found: ${PRETRAINED_PATH}"
  exit 1
fi

if [[ ${CLEAN_ROOT} -eq 1 ]]; then
  echo "Cleaning ${OUTPUT_ROOT}"
  rm -rf -- "${OUTPUT_ROOT}"
fi

TRAIN_ROOT="${OUTPUT_ROOT}/train"
STATUS_ROOT="${OUTPUT_ROOT}/.run_status/seed${SEED}"
SUMMARY_CSV="${OUTPUT_ROOT}/summary_${SOURCE_DATASET}_GRACE_id_ood.csv"
mkdir -p "${TRAIN_ROOT}" "${STATUS_ROOT}"

is_true() {
  case "${1,,}" in true|1|yes|y|on) return 0 ;; *) return 1 ;; esac
}

dataset_for_percent() {
  if [[ "$1" == "100" ]]; then echo "${SOURCE_DATASET}"; else echo "${SOURCE_DATASET}_$1"; fi
}

run_stage() {
  local marker="$1"
  shift
  if is_true "${RESUME}" && [[ -f "${marker}" ]]; then
    echo "[skip] ${marker}"
    return 0
  fi
  "$@"
  date '+%F %T' > "${marker}"
}

echo "========================================"
echo "GRACE ID+OOD"
echo "source=${SOURCE_DATASET} seed=${SEED} gpu=${GPU:-visible-default}"
echo "percents=${PERCENTS[*]} ood_targets=${OOD_TARGETS[*]:-none}"
echo "train_root=${TRAIN_ROOT}"
echo "summary=${SUMMARY_CSV}"
echo "========================================"

for percent in "${PERCENTS[@]}"; do
  dataset_run="$(dataset_for_percent "${percent}")"
  marker="${STATUS_ROOT}/train_${dataset_run}.done"
  train_command=(python train_grace.py)
  latest_checkpoint="${TRAIN_ROOT}/${dataset_run}/trained_models/seed${SEED}/GRACE_UniMedCLIP_ViT-B-16_latest.pth"
  if is_true "${RESUME}" && [[ -f "${latest_checkpoint}" ]]; then
    train_command+=(--resume)
  fi
  train_command+=(
    --config-file "${SOURCE_CONFIG}"
    --output-dir "${TRAIN_ROOT}"
    --data-percentage "${percent}"
    --seed "${SEED}"
    MODEL.UNIMEDCLIP_PRETRAINED_PATH "${PRETRAINED_PATH}"
    "${EXTRA_OPTS[@]}"
  )
  echo "[train] ${SOURCE_DATASET} ${percent}%"
  run_stage "${marker}" "${train_command[@]}"
done

for checkpoint_type in best latest; do
  if [[ "${checkpoint_type}" == "best" ]]; then
    checkpoint_suffix="best_dice"
  else
    checkpoint_suffix="latest"
  fi
  EVAL_ROOT="${OUTPUT_ROOT}/${checkpoint_type}_eval"

  for percent in "${PERCENTS[@]}"; do
    dataset_run="$(dataset_for_percent "${percent}")"
    checkpoint_path="${TRAIN_ROOT}/${dataset_run}/trained_models/seed${SEED}/GRACE_UniMedCLIP_ViT-B-16_${checkpoint_suffix}.pth"
    test_marker="${STATUS_ROOT}/${checkpoint_type}_id_test_${dataset_run}.done"
    eval_marker="${STATUS_ROOT}/${checkpoint_type}_id_eval_${dataset_run}.done"
    echo "[ID/${checkpoint_type}] ${SOURCE_DATASET} ${percent}%"
    run_stage "${test_marker}" \
      python test_grace.py \
        --config-file "${SOURCE_CONFIG}" \
        --output-dir "${EVAL_ROOT}" \
        --source-dataset "${SOURCE_DATASET}" \
        --checkpoint "${checkpoint_path}" \
        --data-percentage "${percent}" \
        --seed "${SEED}" \
        --prompt-design "${PROMPT_DESIGN}" \
        MODEL.UNIMEDCLIP_PRETRAINED_PATH "${PRETRAINED_PATH}" \
        "${EXTRA_OPTS[@]}"
    run_stage "${eval_marker}" \
      python -m utils.eval_grace \
        --config-file "${SOURCE_CONFIG}" \
        --output-dir "${EVAL_ROOT}" \
        --data-percentage "${percent}" \
        --seed "${SEED}" \
        --prompt-design "${PROMPT_DESIGN}" \
        "${EXTRA_OPTS[@]}"
    metrics_csv="${EVAL_ROOT}/${dataset_run}/seg_results/seed${SEED}/test_GRACE_Prompt-${PROMPT_DESIGN}.csv"
    python scripts/record_metrics.py \
      --summary "${SUMMARY_CSV}" \
      --metrics "${metrics_csv}" \
      --checkpoint "${checkpoint_type}" \
      --split id \
      --train-percentage "${percent}" \
      --source-dataset "${SOURCE_DATASET}" \
      --target-dataset "${SOURCE_DATASET}" \
      --checkpoint-path "${checkpoint_path}"
  done

  source_run="$(dataset_for_percent "${OOD_SOURCE_PERCENT}")"
  checkpoint_path="${TRAIN_ROOT}/${source_run}/trained_models/seed${SEED}/GRACE_UniMedCLIP_ViT-B-16_${checkpoint_suffix}.pth"
  for target in "${OOD_TARGETS[@]}"; do
    target_config="configs/${target}_GRACE.yaml"
    if [[ ! -f "${target_config}" ]]; then
      echo "OOD config not found: ${target_config}"
      exit 1
    fi
    test_marker="${STATUS_ROOT}/${checkpoint_type}_ood_test_${SOURCE_DATASET}_${target}.done"
    eval_marker="${STATUS_ROOT}/${checkpoint_type}_ood_eval_${SOURCE_DATASET}_${target}.done"
    echo "[OOD/${checkpoint_type}] ${SOURCE_DATASET} -> ${target}"
    run_stage "${test_marker}" \
      python test_grace.py \
        --config-file "${target_config}" \
        --output-dir "${EVAL_ROOT}" \
        --source-dataset "${SOURCE_DATASET}" \
        --checkpoint "${checkpoint_path}" \
        --data-percentage 100 \
        --seed "${SEED}" \
        --prompt-design "${PROMPT_DESIGN}" \
        MODEL.UNIMEDCLIP_PRETRAINED_PATH "${PRETRAINED_PATH}" \
        "${EXTRA_OPTS[@]}"
    run_stage "${eval_marker}" \
      python -m utils.eval_grace \
        --config-file "${target_config}" \
        --output-dir "${EVAL_ROOT}" \
        --data-percentage 100 \
        --seed "${SEED}" \
        --prompt-design "${PROMPT_DESIGN}" \
        "${EXTRA_OPTS[@]}"
    metrics_csv="${EVAL_ROOT}/${target}/seg_results/seed${SEED}/test_GRACE_Prompt-${PROMPT_DESIGN}.csv"
    python scripts/record_metrics.py \
      --summary "${SUMMARY_CSV}" \
      --metrics "${metrics_csv}" \
      --checkpoint "${checkpoint_type}" \
      --split ood \
      --train-percentage "${OOD_SOURCE_PERCENT}" \
      --source-dataset "${SOURCE_DATASET}" \
      --target-dataset "${target}" \
      --checkpoint-path "${checkpoint_path}"
  done
done

echo "Completed GRACE ID+OOD. Summary: ${SUMMARY_CSV}"
