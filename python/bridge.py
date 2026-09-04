# Python bridge for llm-cpp via numpy-cpp python module (header-only)
# Requires: pip install pybind11 && cmake -DNP_BUILD_PYTHON=ON
try:
    import numpy_cpp  # from numpy-cpp/python
    print("numpy-cpp python available")
except ImportError:
    print("numpy-cpp python not built; using llm-cpp binary via subprocess")
import subprocess, json
def generate(prompt, max_tokens=50):
    out = subprocess.check_output(["./build/llm-cpp", "generate", "--prompt", prompt, "--max_tokens", str(max_tokens)])
    return out.decode()
if __name__ == "__main__":
    print(generate("Hello"))
