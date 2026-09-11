#!/bin/bash
# J97: SBOM + license scan. Fails on GPL text in any dependency; warns on missing.
set -u
cd "$(dirname "$0")/.."
PIN_NP=$(git -C build/_deps/numpy-cpp-src rev-parse HEAD 2>/dev/null || echo "unpinned")
if [ -d /home/sergio/Project/numpy-cpp/.git ]; then
  PIN_LOCAL=$(git -C /home/sergio/Project/numpy-cpp rev-parse HEAD 2>/dev/null || echo "?")
else
  PIN_LOCAL="n/a"
fi
echo "== SBOM =="
echo "- llm-cpp (MIT, LICENSE)"
echo "- numpy-cpp FetchContent@$(echo "$PIN_NP" | cut -c1-12) local@$(echo "$PIN_LOCAL" | cut -c1-12)"
echo "- system: libstdc++, libm, libpthread, OpenMP (runtime, if enabled)"
echo ""
echo "== license scan =="
FAIL=0
for d in build/_deps/numpy-cpp-src /home/sergio/Project/numpy-cpp; do
  [ -d "$d" ] || continue
  if grep -ril "general public license" "$d" --include="*.h" --include="*.hpp" --include="*.cpp" --include="*.txt" --include="*.md" 2>/dev/null | head -n 3 | grep -q .; then
    echo "GPL FOUND in $d (forbidden)"; FAIL=1
  else
    echo "no GPL text in $d"
  fi
  if [ -f "$d/LICENSE" ] || [ -f "$d/LICENSE.md" ] || [ -f "$d/COPYING" ]; then
    echo "license file present in $d"
  else
    echo "WARNING: no LICENSE file in $d (recorded exception, see THIRD_PARTY.md)"
  fi
done
[ "$FAIL" = "0" ] && echo "sbom clean" || { echo "sbom FAILED"; exit 1; }
