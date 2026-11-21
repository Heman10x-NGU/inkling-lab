import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from official import get
def blockq(W,bits,block=32):
    """Symmetric per-block absmax quantizer (the simple Q4_0/Q8_0 family). Returns reconstruction."""
    r,c=W.shape; X=W.reshape(r,c//block,block); m=np.abs(X).max(-1,keepdims=True); q=2**(bits-1)-1
    return (np.round(X/np.maximum(m,1e-12)*q)/q*m).reshape(r,c)
print("relative error of attention matrices at lower bit widths (layer 8, official weights, simple block quantizer)")
print("matrix       8-bit    6-bit    5-bit    4-bit    3-bit   | bytes at 8-bit -> 4-bit per layer")
for m in ["wq_du","wo_ud","gate_attn","wk_dv","wv_dv","wr_du"]:
    W=get(f"model.llm.layers.8.attn.{m}.weight").astype(np.float32); n=np.linalg.norm(W); row=[]
    for b in [8,6,5,4,3]: row.append(np.linalg.norm(blockq(W,b)-W)/n)
    print(f"{m:10s} "+" ".join(f"{x:8.4f}" for x in row)+f"   | {W.size*8.5/8/1e6:5.2f} MB -> {W.size*4.5/8/1e6:5.2f} MB")
