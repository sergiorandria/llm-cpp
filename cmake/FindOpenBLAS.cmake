find_path(OpenBLAS_INCLUDE_DIR cblas.h
    PATHS /usr/include /usr/include/openblas /usr/local/include /opt/openblas/include)
find_library(OpenBLAS_LIBRARY NAMES openblas
    PATHS /usr/lib /usr/lib64 /usr/local/lib /opt/openblas/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS DEFAULT_MSG OpenBLAS_LIBRARY OpenBLAS_INCLUDE_DIR)

if(OpenBLAS_FOUND)
    set(OpenBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIR})
    set(OpenBLAS_LIBRARIES ${OpenBLAS_LIBRARY})
    message(STATUS "Found OpenBLAS: ${OpenBLAS_LIBRARY} at ${OpenBLAS_INCLUDE_DIR} — Tensor::matmul will use cblas_sgemm when -DUSE_OPENBLAS=ON")
else()
    message(STATUS "OpenBLAS not found — Tensor::matmul fallback to blocked GEMM. Install libopenblas-dev for cblas_sgemm")
endif()
mark_as_advanced(OpenBLAS_INCLUDE_DIR OpenBLAS_LIBRARY)
