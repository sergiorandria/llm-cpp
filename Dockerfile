FROM ubuntu:22.04
RUN apt-get update && apt-get install -y cmake g++ make libboost-all-dev
WORKDIR /app
COPY . .
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ENTRYPOINT ["./build/llm-cpp"]
