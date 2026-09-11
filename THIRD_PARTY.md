# Third-party dependencies (J97)

| Dep | Pin | License | Notes |
|-----|-----|---------|-------|
| llm-cpp itself | — | MIT (`LICENSE`) | — |
| numpy-cpp (FetchContent `main`) | `02a2384` (build/_deps) | **claimed BSD-3-Clause** (README badge) | ⚠️ NO LICENSE file in repo or upstream HEAD — exception recorded; follow-up: request upstream LICENSE before 1.0 |
| system (libstdc++, libm, pthread, OpenMP, curl in image) | distro | GPL-with-exception / BSD | runtime only |

`scripts/sbom.sh` fails CI on GPL text in any dep; missing-license warns (see exception above).
