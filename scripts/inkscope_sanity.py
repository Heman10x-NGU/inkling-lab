"""inkscope_sanity.py <capture_dir> <id> : result_norm x dequantized output.weight (with the 30 softcap) vs the tool's top-5."""
import sys, json, numpy as np, gguf
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from inkscope_read import records
M=str(pathlib.Path(__file__).resolve().parents[1] / "models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf")
d, pid = sys.argv[1], sys.argv[2]
h = [a for n,_,_,a in records(f"{d}/{pid}.bin") if n == "result_norm"][-1].reshape(-1, 2304)[-1].astype(np.float64)
r = gguf.GGUFReader(M); t = next(x for x in r.tensors if x.name == "output.weight")
W = gguf.quants.dequantize(t.data, t.tensor_type).reshape(-1, 2304)    # numpy (n_vocab_padded, 2304)
z = 30*np.tanh((W @ h)/30)[:200058]
top = np.argsort(-z)[:5]; js = json.load(open(f"{d}/{pid}.json"))
print("numpy top5:", top.tolist(), np.round(z[top],3).tolist())
print("tool  top5:", [e["id"] for e in js["top5"]], [round(e["logit"],3) for e in js["top5"]])
print("top-1 match:", int(top[0]) == js["top5"][0]["id"], " max |dlogit| on tool top5:", max(abs(z[e["id"]]-e["logit"]) for e in js["top5"]))
