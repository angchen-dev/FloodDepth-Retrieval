#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
RUN_ARGS=("$@")
SUMMARY_ARGS=()
DRY_RUN=false
SHOW_HELP=false

# Forward only the options shared by the experiment runner and summarizer.
index=0
while (( index < ${#RUN_ARGS[@]} )); do
    argument="${RUN_ARGS[index]}"
    case "${argument}" in
        --runs-dir)
            SUMMARY_ARGS+=("${argument}" "${RUN_ARGS[index + 1]}")
            ((index += 2))
            continue
            ;;
        --runs-dir=*)
            SUMMARY_ARGS+=("${argument}")
            ;;
        --t-values|--area-weights)
            SUMMARY_ARGS+=("${argument}")
            ((index += 1))
            while (( index < ${#RUN_ARGS[@]} )) && [[ "${RUN_ARGS[index]}" != --* ]]; do
                SUMMARY_ARGS+=("${RUN_ARGS[index]}")
                ((index += 1))
            done
            continue
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        -h|--help)
            SHOW_HELP=true
            ;;
    esac
    ((index += 1))
done

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/ablation/run_ablation.py "${RUN_ARGS[@]}"

if [[ "${SHOW_HELP}" == true ]]; then
    exit 0
fi

if [[ "${DRY_RUN}" == true ]]; then
    printf '+ %q' "${PYTHON_BIN}" scripts/ablation/summarize_ablation.py "${SUMMARY_ARGS[@]}"
    printf '\n'
else
    "${PYTHON_BIN}" scripts/ablation/summarize_ablation.py "${SUMMARY_ARGS[@]}"
fi
