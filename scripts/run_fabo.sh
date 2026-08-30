#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <nets-file> <out-prefix> [tier] [limit]" >&2
  echo "tiers: t3, t7, t8, t14" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NETS="$1"
OUT_PREFIX="$2"
TIER="${3:-t3}"
LIMIT="${4:-}"

case "$OUT_PREFIX" in
  /*) ;;
  *) OUT_PREFIX="$ROOT/$OUT_PREFIX" ;;
esac

cmake -S "$ROOT/src" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$ROOT/build" --target run_fabo -j 8 >/dev/null

clear_salt_env() {
  while IFS='=' read -r k _; do
    case "$k" in
      FAR_*|SALT_*)
        unset "$k"
        ;;
    esac
  done < <(env)
}

clear_salt_env

case "$TIER" in
  t3)  CONFIG="$ROOT/configs/tier_t3_fast.env" ;;
  t7)  CONFIG="$ROOT/configs/tier_t7_yx.env" ;;
  t8)  CONFIG="$ROOT/configs/tier_t8_anchor7.env" ;;
  t14) CONFIG="$ROOT/configs/tier_t14_quality.env" ;;
  *)
    echo "unknown tier '$TIER' (expected: t3, t7, t8, t14)" >&2
    exit 2
    ;;
esac

set -a
source "$CONFIG"
set +a

mkdir -p "$(dirname "$OUT_PREFIX")"

args=(
  -nets "$NETS"
  -out "${OUT_PREFIX}.csv"
  --out-cands "${OUT_PREFIX}_cands.csv"
  --methods SALT_R3,FAR_SALT_FULL
  --eps 0.5
  --wt-max 1.03
)

if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi

(cd "$ROOT" && "$ROOT/build/run_fabo" "${args[@]}")

echo "FABO operating point: $TIER"
echo "wrote ${OUT_PREFIX}.csv"
echo "wrote ${OUT_PREFIX}_cands.csv"
