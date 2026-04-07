#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SKILL_PATH="${PROJECT_ROOT}/.agents/skills/gdn-optimization-workflow/SKILL.md"
DECODE_TEMPLATE="${PROJECT_ROOT}/.agents/skills/gdn-optimization-workflow/references/decode-template.md"
PREFILL_TEMPLATE="${PROJECT_ROOT}/.agents/skills/gdn-optimization-workflow/references/prefill-template.md"

usage() {
  cat <<'EOF'
Usage:
  scripts/ralph_codex_opt.sh <decode|prefill> [topic...]
  scripts/ralph_codex_opt.sh <decode|prefill> [topic...] --model <model>
  scripts/ralph_codex_opt.sh <decode|prefill> [topic...] --max-iterations <n>
  scripts/ralph_codex_opt.sh <decode|prefill> [topic...] --sleep <seconds>
  scripts/ralph_codex_opt.sh <decode|prefill> [topic...] --json
  scripts/ralph_codex_opt.sh <decode|prefill> [topic...] --output <file>

Description:
  Runs the GDN optimization workflow in a Ralph-style loop using Codex headless
  mode and the project-local skill. By default it repeats until interrupted.

Examples:
  scripts/ralph_codex_opt.sh decode
  scripts/ralph_codex_opt.sh decode warp reduction
  scripts/ralph_codex_opt.sh decode --max-iterations 5
  scripts/ralph_codex_opt.sh prefill shared memory tiling --model gpt-5.4
  scripts/ralph_codex_opt.sh decode --json --output /tmp/decode-last.txt
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage
  exit 0
fi

TARGET="$1"
shift

if [[ "${TARGET}" != "decode" && "${TARGET}" != "prefill" ]]; then
  echo "error: target must be 'decode' or 'prefill'" >&2
  usage
  exit 1
fi

MODEL=""
JSON_MODE=0
OUTPUT_FILE=""
MAX_ITERATIONS=0
SLEEP_SECS=0
TOPIC_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || { echo "error: --model requires a value" >&2; exit 1; }
      MODEL="$2"
      shift 2
      ;;
    --json)
      JSON_MODE=1
      shift
      ;;
    --max-iterations)
      [[ $# -ge 2 ]] || { echo "error: --max-iterations requires a value" >&2; exit 1; }
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --sleep)
      [[ $# -ge 2 ]] || { echo "error: --sleep requires a value" >&2; exit 1; }
      SLEEP_SECS="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || { echo "error: --output requires a value" >&2; exit 1; }
      OUTPUT_FILE="$2"
      shift 2
      ;;
    *)
      TOPIC_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "${SKILL_PATH}" ]]; then
  echo "error: missing skill file: ${SKILL_PATH}" >&2
  exit 1
fi

if [[ "${TARGET}" == "decode" ]]; then
  TEMPLATE_PATH="${DECODE_TEMPLATE}"
else
  TEMPLATE_PATH="${PREFILL_TEMPLATE}"
fi

if [[ ! -f "${TEMPLATE_PATH}" ]]; then
  echo "error: missing template file: ${TEMPLATE_PATH}" >&2
  exit 1
fi

if ! [[ "${MAX_ITERATIONS}" =~ ^[0-9]+$ ]]; then
  echo "error: --max-iterations must be a non-negative integer" >&2
  exit 1
fi

if ! [[ "${SLEEP_SECS}" =~ ^[0-9]+$ ]]; then
  echo "error: --sleep must be a non-negative integer" >&2
  exit 1
fi

TOPIC_TEXT=""
if [[ ${#TOPIC_ARGS[@]} -gt 0 ]]; then
  TOPIC_TEXT="${TOPIC_ARGS[*]}"
fi

build_prompt() {
  local iteration="$1"
  cat <<EOF
Use the project-local Codex skill:
[$(basename "$(dirname "${SKILL_PATH}")")](${SKILL_PATH})

Run exactly one optimization iteration for the \`${TARGET}\` GDN kernel in this repository.
Follow the workflow in the skill and in \`AGENTS.md\`.

Constraints:
- Use the latest comparable logged quick benchmark result as baseline.
- Use quick Modal benchmark only.
- Implement exactly one optimization idea.
- Review the kernel before benchmarking.
- Revert if correctness regresses or speedup does not improve.
- Commit only if correctness is preserved and performance improves.

Loop context:
- Ralph loop iteration: ${iteration}
- This script will invoke Codex again after this iteration completes unless stopped.
- Finish this iteration cleanly and report the decision clearly.

Selected target: ${TARGET}
Optional topic: ${TOPIC_TEXT:-none}

Use this runbook while working:
\`${TEMPLATE_PATH}\`
EOF
}

ITERATION=1
while true; do
  PROMPT="$(build_prompt "${ITERATION}")"

  CMD=(
    codex
    --ask-for-approval never
    exec
    --cd "${PROJECT_ROOT}"
    --sandbox danger-full-access
  )

  if [[ -n "${MODEL}" ]]; then
    CMD+=(--model "${MODEL}")
  fi

  if [[ ${JSON_MODE} -eq 1 ]]; then
    CMD+=(--json)
  fi

  if [[ -n "${OUTPUT_FILE}" ]]; then
    if (( MAX_ITERATIONS > 0 )); then
      base="${OUTPUT_FILE}"
      ext=""
      if [[ "${OUTPUT_FILE}" == *.* ]]; then
        base="${OUTPUT_FILE%.*}"
        ext=".${OUTPUT_FILE##*.}"
      fi
      CMD+=(--output-last-message "${base}.iter-${ITERATION}${ext}")
    else
      CMD+=(--output-last-message "${OUTPUT_FILE}")
    fi
  fi

  CMD+=("${PROMPT}")

  echo "Running Ralph loop iteration ${ITERATION} for target='${TARGET}' topic='${TOPIC_TEXT:-none}'" >&2
  "${CMD[@]}"

  if (( MAX_ITERATIONS > 0 && ITERATION >= MAX_ITERATIONS )); then
    echo "Ralph loop completed ${ITERATION} iteration(s); stopping due to --max-iterations." >&2
    break
  fi

  ((ITERATION++))

  if (( SLEEP_SECS > 0 )); then
    echo "Sleeping ${SLEEP_SECS}s before next Ralph loop iteration." >&2
    sleep "${SLEEP_SECS}"
  fi
done
