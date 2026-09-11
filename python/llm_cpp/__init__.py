"""G63: ctypes bindings for libllm-c-api (no pybind11 needed)."""
import ctypes
import os

_lib = None


def _libpath():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in [
        os.path.join(here, "..", "build", "libllm-c-api.so"),
        os.path.join(os.getcwd(), "build", "libllm-c-api.so"),
        "libllm-c-api.so",
    ]:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("libllm-c-api.so not found (build with cmake first)")


def load(path=None):
    global _lib
    lib = ctypes.CDLL(path or _libpath())
    lib.llm_create.argtypes = [ctypes.c_size_t] * 5
    lib.llm_create.restype = ctypes.c_void_p
    lib.llm_free.argtypes = [ctypes.c_void_p]
    lib.llm_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                               ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.llm_encode.restype = ctypes.c_int
    lib.llm_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                               ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.llm_decode.restype = ctypes.c_int
    lib.llm_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.llm_generate.restype = ctypes.c_int
    lib.llm_num_parameters.argtypes = [ctypes.c_void_p]
    lib.llm_num_parameters.restype = ctypes.c_size_t
    _lib = lib
    return lib


class Model:
    def __init__(self, vocab=32, layers=1, heads=2, embd=8, block=16, path=None):
        lib = load(path)
        self._lib = lib
        self._m = lib.llm_create(vocab, layers, heads, embd, block)
        if not self._m:
            raise ValueError("invalid config")

    def __del__(self):
        if getattr(self, "_m", None):
            self._lib.llm_free(self._m)
            self._m = None

    def encode(self, text):
        b = text.encode("utf-8")
        n = self._lib.llm_encode(self._m, b, None, 0)
        out = (ctypes.c_int * n)()
        self._lib.llm_encode(self._m, b, out, n)
        return list(out)

    def decode(self, ids):
        arr = (ctypes.c_int * len(ids))(*ids)
        n = self._lib.llm_decode(self._m, arr, len(ids), None, 0)
        buf = ctypes.create_string_buffer(n)
        self._lib.llm_decode(self._m, arr, len(ids), buf, n)
        return buf.raw.decode("utf-8")

    def generate(self, prompt, max_tokens=10):
        n = self._lib.llm_generate(self._m, prompt.encode("utf-8"), max_tokens, None, 0)
        out = (ctypes.c_int * n)()
        self._lib.llm_generate(self._m, prompt.encode("utf-8"), max_tokens, out, n)
        return list(out)

    def num_parameters(self):
        return self._lib.llm_num_parameters(self._m)
