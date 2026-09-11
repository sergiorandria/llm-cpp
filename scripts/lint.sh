#!/bin/bash
# J93: format + tidy gate on CHANGED files (legacy grandfathered; new code must be clean).
set -e
cd "$(dirname "$0")/.."
BASE="${1:-}"
if [ -z "$BASE" ]; then
  if git rev-parse --verify origin/main > /dev/null 2>&1; then
    BASE=$(git merge-base HEAD origin/main)
  else
    BASE="HEAD"
  fi
fi
FILES=$(git diff --name-only "$BASE" HEAD -- 'src/*.cpp' 'include/llm/*.h' 'tests/*.cpp' 'tests/*.h' || true)
# plus uncommitted changes
FILES="$FILES $(git status --porcelain -- 'src/*.cpp' 'include/llm/*.h' 'tests/*.cpp' 'tests/*.h' | awk '{print $2}')"
# generated headers are autotune output, not hand code
FILES=$(echo "$FILES" | tr ' ' '\n' | sort -u | grep -E '\.(cpp|h)$' | grep -vE '(flash_config|gemm_config)\.h$' || true)
if [ -z "$FILES" ]; then echo "lint: no changed llm sources, skipping"; exit 0; fi
echo "== clang-format (changed files) =="
echo "$FILES"
echo "$FILES" | xargs clang-format --dry-run --Werror
echo "format clean"
# clang-tidy is slow on numpy-heavy TUs: opt-in via LINT_TIDY=1 (CI sets it).
if [ "${LINT_TIDY:-0}" = "1" ]; then
echo "== clang-tidy =="
if [ ! -f build/compile_commands.json ]; then
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON > /dev/null
fi
echo "$FILES" | grep '\.cpp$' | xargs clang-tidy -p build --quiet || {
  echo "tidy findings above (advisory)";
}
fi
echo "lint done"
