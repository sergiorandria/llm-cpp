#!/usr/bin/env python3
import time, subprocess
start=time.time()
subprocess.run(["./build/llm-cpp"], check=False)
print(f"benchmark: {time.time()-start:.2f}s")
