import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from official import get, header
from concurrent.futures import ThreadPoolExecutor
H=header()["tensors"]; np.set_printoptions(precision=3,suppress=True,linewidth=200)
def g(n): return get(n).astype(np.float32)
small=[k for k,v in H.items() if "visual" not in k and "experts" not in k and (v["data_offsets"][1]-v["data_offsets"][0])<3_000_000]
with ThreadPoolExecutor(8) as ex: list(ex.map(get,small))
print("cached small tensors:",len(small))
# (a) gate_attn dominant direction
print("\n(a) gate_attn top singular direction: sigma1/sigma2 | top-3 input dims (|v| coordinate) | left-vector uniformity (|mean u|*sqrt(n))")
for L in [2,5,8,11,14,17,20,23]:
    W=g(f"model.llm.layers.{L}.attn.gate_attn.weight"); U,S,Vt=np.linalg.svd(W,full_matrices=False)
    v=Vt[0]; idx=np.argsort(-np.abs(v))[:3]; u=U[:,0]
    print(f"L{L:02d} s1/s2={S[0]/S[1]:5.1f} top dims {[(int(i),round(float(v[i]),3)) for i in idx]} | energy in top-10 dims {float((np.sort(v**2)[-10:]).sum()):.3f} | left uniform {abs(u.mean())*np.sqrt(len(u)):.3f} | row-mean-vector norm {np.linalg.norm(W.mean(0)):.3f}")
# (b) privileged basis: norm weights and outlier dims
print("\n(b) per-dimension norm weights: top outlier dims (dim, value/median)")
for name in ["model.llm.embed_norm.weight","model.llm.norm.weight"]+[f"model.llm.layers.{L}.attn_norm.weight" for L in [0,6,11,16,17,18]]+[f"model.llm.layers.{L}.mlp_norm.weight" for L in [0,6,11,17,18,23]]:
    if name not in H: print(name.split("llm.")[1],"absent"); continue
    w=g(name); med=np.median(np.abs(w)); idx=np.argsort(-np.abs(w))[:4]
    print(f"{name.split('llm.')[1]:28s} median {med:.3f} min {w.min():.3f} max {w.max():.3f} top {[(int(i),round(float(w[i]/med),1)) for i in idx]} w[2027]/med={w[2027]/med:.2f}")
# which residual dims are read hardest by attention/gates: column norms
print("\n(b2) column (input-dim) norm outliers: matrix | max col norm / median | argmax dim | col 2027 / median")
for L in [5,11,17,23]:
    for m in ["wq_du","wk_dv","wv_dv","gate_attn","wr_du"]:
        W=g(f"model.llm.layers.{L}.attn.{m}.weight"); c=np.linalg.norm(W,axis=0); med=np.median(c)
        print(f"L{L:02d} {m:10s} max/med {c.max()/med:6.2f} argmax {int(c.argmax()):5d} | col2027/med {c[2027]/med:6.2f}")
    W=g(f"model.llm.layers.{L}.mlp.gate.weight"); c=np.linalg.norm(W,axis=0); med=np.median(c); print(f"L{L:02d} router     max/med {c.max()/med:6.2f} argmax {int(c.argmax()):5d} | col2027/med {c[2027]/med:6.2f}")
# (c) controllers: how much each gate listens to current stream x vs message m
print("\n(c) loop controllers: fraction of down_weight energy on first half (x) vs second half (m) | forget/gain row-norm outlier dim")
for L in range(6,18):
    row=[]
    for c in ["attn","k","v","mlp"]:
        D=g(f"model.llm.layers.{L}.controllers.{c}.down_weight"); e1=(D[:,:2304]**2).sum(); e2=(D[:,2304:]**2).sum()
        F=g(f"model.llm.layers.{L}.controllers.{c}.forget_weight"); Gn=g(f"model.llm.layers.{L}.controllers.{c}.gain_weight")
        fn=np.linalg.norm(F,axis=1); gn=np.linalg.norm(Gn,axis=1)
        row.append(f"{c}: x={e1/(e1+e2):.2f} F[{int(fn.argmax())}]x{fn.max()/np.median(fn):.0f} G[{int(gn.argmax())}]x{gn.max()/np.median(gn):.0f}")
    print(f"L{L:02d} "+" | ".join(row))
# (d) cross-pass queries
for n in ["model.llm.middle_cycle_cross_pass_queries","model.llm.middle_cycle_cross_pass_kv_queries"]:
    Q=g(n); nr=np.linalg.norm(Q,axis=1); print(f"\n(d) {n.split('llm.')[1]} row norms:",nr.round(2)); idx=np.argsort(-np.abs(Q).mean(0))[:5]; print("   dims with largest mean |value|:",[(int(i),round(float(np.abs(Q).mean(0)[i]/np.median(np.abs(Q).mean(0))),1)) for i in idx])
# (e) attention-residual queries
print("\n(e) attn_res queries: shapes and norms")
for k in sorted([k for k in H if "attn_res" in k],key=lambda s:(s.split('.')[-2],int(s.split('.')[-1]) if s.split('.')[-1].isdigit() else -1)):
    q=g(k); print(f"   {k.split('attn_res.')[1]:18s} shape {list(q.shape)} norm {np.atleast_2d(np.linalg.norm(np.atleast_2d(q),axis=1)).round(2).tolist()[0]} max|dim| {int(np.abs(np.atleast_2d(q)).max(0).argmax())}")
# (f) short convolutions over the last 4 tokens
print("\n(f) depthwise token convolutions (kernel 4): mean |weight| per tap, and fraction of dims whose largest tap is not the last one")
for L in [0,3,6,9,12,15,17,20,23]:
    for nm in ["attn_sconv","mlp_sconv"]:
        K=g(f"model.llm.layers.{L}.{nm}.weight").reshape(2304,4); a=np.abs(K).mean(0); big=np.abs(K).argmax(1)
        print(f"L{L:02d} {nm:10s} taps mean|w| {a.round(3)} signed mean {K.mean(0).round(3)} argmax-tap histogram {np.bincount(big,minlength=4).tolist()}")
