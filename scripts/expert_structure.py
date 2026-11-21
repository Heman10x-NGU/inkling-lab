import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from official import get, get_raw, decode_fp4, header, LAB
L=int(sys.argv[1]); H=header()["tensors"]
n13=f"model.llm.layers.{L}.mlp.experts.w13_weight"; v=H[n13]; E,R,Cp=v["shape"]; nb=v["data_offsets"][1]-v["data_offsets"][0]; C=H[n13+"_scale"]["shape"][2]*16
print(f"layer {L} w13 stored shape {v['shape']} logical [{E},{R},{C}] bytes {nb} -> bits/elem {8*nb/(E*R*C):.2f}; scale shape {H[n13+'_scale']['shape']}")
assert abs(8*nb/(E*R*C)-4)<1e-6, "not a 4-bit layer"
raw=get_raw(n13); sc=get_raw(n13+"_scale"); per=R*C//2; sper=R*(C//16)
def expert(e): return decode_fp4(raw[e*per:(e+1)*per], sc[e*sper:(e+1)*sper], R, C)
W0=expert(0); rn=np.linalg.norm(W0,axis=1)
print(f"decode check, expert 0: row-norm mean {rn.mean():.4f} std/mean {rn.std()/rn.mean():.4f} (near 0 means rows are equal-norm, as in the other matrices); frac zeros {float((W0==0).mean()):.3f}; value std {W0.std():.4f}")
# 1) rank of individual experts
print("\nsingle-expert spectra (w13 = gate+up, 2304x2304): expert | stable rank | r90 | r99 | r90 of gate half | r90 of up half")
res=[]
for e in [0,17,45,77,101,127]:
    W=expert(e); s=np.linalg.svd(W,compute_uv=False); c=np.cumsum(s**2)/np.sum(s**2)
    sg=np.linalg.svd(W[:R//2],compute_uv=False); cg=np.cumsum(sg**2)/np.sum(sg**2); su=np.linalg.svd(W[R//2:],compute_uv=False); cu=np.cumsum(su**2)/np.sum(su**2)
    r=dict(e=e,stable=float((s**2).sum()/s[0]**2),r90=int(np.searchsorted(c,.9)+1),r99=int(np.searchsorted(c,.99)+1),g90=int(np.searchsorted(cg,.9)+1),u90=int(np.searchsorted(cu,.9)+1)); res.append(r)
    print(f"  e{e:03d} stable {r['stable']:7.1f} r90 {r['r90']:5d} r99 {r['r99']:5d} | gate r90 {r['g90']:4d}/1152 | up r90 {r['u90']:4d}/1152")
# 2) shared structure across experts: Gram of normalized experts, mean-expert energy
X=np.empty((E,R*C),dtype=np.float16); nrm=np.empty(E,dtype=np.float64); mean=np.zeros(R*C,dtype=np.float64)
for e in range(E):
    w=expert(e).reshape(-1); n=np.linalg.norm(w); nrm[e]=n; X[e]=(w/n).astype(np.float16); mean+=w/n
mean/=E; G=np.zeros((E,E),dtype=np.float64); step=400_000
for o in range(0,R*C,step):
    Z=X[:,o:o+step].astype(np.float32); G+=Z@Z.T
ev=np.linalg.eigvalsh(G)[::-1]; off=G[~np.eye(E,dtype=bool)]
print(f"\ncross-expert weight similarity (cosine of whole matrices): mean {off.mean():+.4f} max {off.max():+.4f} min {off.min():+.4f}")
print(f"mean-expert energy fraction: {float((mean**2).sum()):.4f} (1/128 = {1/128:.4f} if experts were unrelated)")
print(f"Gram eigenvalues: top5 {ev[:5].round(3).tolist()} ; participation ratio {float(ev.sum()**2/(ev**2).sum()):.1f} of {E}")
print(f"expert norm spread: min {nrm.min():.2f} max {nrm.max():.2f}")
# 3) shared input subspace: do experts read the same directions? column-energy profile correlation
ce=np.stack([ (X[e].astype(np.float32).reshape(R,C)**2).sum(0) for e in range(0,E,8)])
cc=np.corrcoef(ce); print(f"input-dimension energy profile correlation between experts (16 sampled): mean off-diag {cc[~np.eye(len(cc),dtype=bool)].mean():.3f}")
tot=ce.mean(0); idx=np.argsort(-tot)[:8]; print("input dims the experts read hardest (dim, x median):",[(int(i),round(float(tot[i]/np.median(tot)),1)) for i in idx], "| dim 2027:",round(float(tot[2027]/np.median(tot)),2))
json.dump(dict(layer=L,spectra=res,offdiag_mean=float(off.mean()),offdiag_max=float(off.max()),mean_energy=float((mean**2).sum()),pr=float(ev.sum()**2/(ev**2).sum())),open(os.path.join(LAB,"results",f"expert_structure_L{L}.json"),"w"),indent=1)
