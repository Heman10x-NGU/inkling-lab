"""inkscope_wo_check.py <capture_dir> <id> : conv(W_o @ wo_inp) == attn_out, i.e. wo_inp feature j is input column j of attn_output.weight."""
import sys, numpy as np, gguf
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from inkscope_read import records
M=str(pathlib.Path(__file__).resolve().parents[1] / "models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf")
R = {n: a for n,_,_,a in records(f"{sys.argv[1]}/{sys.argv[2]}.bin")}
r = gguf.GGUFReader(M); T = {t.name: t for t in r.tensors}
deq = lambda n: gguf.quants.dequantize(T[n].data, T[n].tensor_type)
for il, L in ((0,0),(20,8),(47,23)):
    if f"attn_out-{il}" not in R: print(f"visit {il}: attn_out not captured"); continue
    k = R[f"wo_inp-{il}"]; a = R[f"attn_out-{il}"]               # numpy (tokens, 3072), (tokens, 2304)
    W = deq(f"blk.{L}.attn_output.weight").reshape(2304, 3072)   # ne [3072,2304] -> numpy (out, in)
    cw = deq(f"blk.{L}.attn_post_conv.weight").reshape(2304, 4)  # ne [4,2304]
    y = k @ W.T; n = y.shape[0]; yp = np.vstack([np.zeros((3, 2304)), y])
    out = (y + sum(cw[:, j] * yp[j:j+n] for j in range(4))) / np.sqrt(5)   # convolve(): (x + conv(x))/sqrt(2*0+4+1)
    print(f"visit {il} (layer {L}): wo_inp {k.shape} rel err conv(W k) vs attn_out = {np.linalg.norm(out-a)/np.linalg.norm(a):.2e}")
