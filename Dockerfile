# G64: production image — multi-stage, non-root, minimal runtime.
ARG UBUNTU=24.04
FROM ubuntu:$UBUNTU AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    g++ cmake make python3 git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DUSE_NUMPY_CPP=ON && \
    cmake --build build -j"$(nproc)" --target llm-cpp && \
    ./build/llm-cpp --help

FROM ubuntu:$UBUNTU AS runtime
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl libgomp1 && rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 10001 llm
COPY --from=build /src/build/llm-cpp /usr/local/bin/llm-cpp
COPY --from=build /src/config /home/llm/config
USER llm
WORKDIR /home/llm
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -sf http://127.0.0.1:8080/healthz || exit 1
ENTRYPOINT ["llm-cpp", "serve", "--port", "8080"]
