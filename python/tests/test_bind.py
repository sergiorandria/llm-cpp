"""G63: binding smoke — encode/decode roundtrip, generate length, params>0."""
import sys
sys.path.insert(0, "python")
from llm_cpp import Model

m = Model(vocab=32, layers=1, heads=2, embd=8, block=16)
assert m.num_parameters() > 0, "params must be >0"
ids = m.encode("Hello")
assert m.decode(ids) == "Hello", "roundtrip must hold"
gen = m.generate("Hi", 5)
prompt = m.encode("Hi")
assert len(gen) == len(prompt) + 5, f"expected prompt+5, got {len(gen)}"
print(f"bind ok params={m.num_parameters()} gen_len={len(gen)}")
