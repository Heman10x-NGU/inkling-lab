import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from official import get, LAB
from concurrent.futures import ThreadPoolExecutor
LAYERS=[2,5,8,11,14,17,20,23]; MATS=["wq_du","wk_dv","wv_dv","wo_ud","gate_attn","wr_du"]
names=[f"model.llm.layers.{L}.attn.{m}.weight" for L in LAYERS for m in MATS]
with ThreadPoolExecutor(6) as ex: list(ex.map(get,names))
def spec(W):
    s=np.linalg.svd(W.astype(np.float32),compute_uv=False); e=s**2; c=np.cumsum(e)/e.sum()
    return dict(full=int(min(W.shape)), stable=float(e.sum()/e[0]), r90=int(np.searchsorted(c,0.90)+1), r99=int(np.searchsorted(c,0.99)+1), top=float(s[0]), med=float(np.median(s)), fro=float(np.sqrt(e.sum())))
out={}
print("layer mat        full  stable  r90   r99   r90/full  s_max/s_med")
for L in LAYERS:
    for m in MATS:
        W=get(f"model.llm.layers.{L}.attn.{m}.weight"); r=spec(W); out[f"{L}.{m}"]=r
        print(f"L{L:02d}  {m:10s} {r['full']:5d} {r['stable']:7.1f} {r['r90']:5d} {r['r99']:5d}   {r['r90']/r['full']:.2f}     {r['top']/max(r['med'],1e-9):7.1f}")
json.dump(out,open(os.path.join(LAB,"results","attn_spectra.json"),"w"),indent=1)
# per-head structure in one core global layer and one local layer
for L in [8,11]:
    Wq=get(f"model.llm.layers.{L}.attn.wq_du.weight").reshape(24,128,2304); Wk=get(f"model.llm.layers.{L}.attn.wk_dv.weight").reshape(8,128,2304)
    Wo=get(f"model.llm.layers.{L}.attn.wo_ud.weight"); G=get(f"model.llm.layers.{L}.attn.gate_attn.weight")
    hn=np.linalg.norm(Wq,axis=(1,2)); kn=np.linalg.norm(Wk,axis=(1,2))
    print(f"\nL{L:02d} query-head norms: min {hn.min():.2f} max {hn.max():.2f} | kv-head norms: min {kn.min():.2f} max {kn.max():.2f}")
    # subspace overlap between query heads: principal-angle energy
    Q=[np.linalg.qr(Wq[h].T)[0] for h in range(24)]
    ov=np.array([[np.linalg.norm(Q[a].T@Q[b])**2/128 for b in range(24)] for a in range(24)]); np.fill_diagonal(ov,0)
    print(f"L{L:02d} query-head input-subspace overlap (1=same subspace, chance={128/2304:.3f}): mean {ov.sum()/(24*23):.3f} max {ov.max():.3f}")
    print(f"L{L:02d} gate_attn: mean {G.mean():+.4f} std {G.std():.4f} | wo_ud std {Wo.std():.4f} | wq std {Wq.std():.4f}")
