import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from official import get, get_raw, decode_fp4, header, LAB
from ggufmap import index
import gguf
from gguf import quants, GGMLQuantizationType as QT
L=int(sys.argv[1]); H=header()["tensors"]
kv,T,mm=index(os.path.join(LAB,"models/atomicchat/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf"))
def gtensor(name):
    t=T[name]; raw=np.frombuffer(mm,dtype=np.uint8,count=t["nbytes"],offset=t["off"]); return t,raw
# attention: Q8_0 in the file vs BF16 official
print("attention matrices, 8-bit file vs official (cosine of whole matrix):")
for gname,oname in [("attn_q","wq_du"),("attn_k","wk_dv"),("attn_v","wv_dv"),("attn_output","wo_ud"),("attn_gate","gate_attn")]:
    t,raw=gtensor(f"blk.{L}.{gname}.weight"); shape=tuple(reversed(t["dims"]))
    A=quants.dequantize(raw.reshape(shape[0],-1),QT(t["type"])).reshape(shape); B=get(f"model.llm.layers.{L}.attn.{oname}.weight").astype(np.float32)
    c=float((A*B).sum()/np.linalg.norm(A)/np.linalg.norm(B)); print(f"  {gname:12s} cos {c:.6f} relative error {np.sqrt(max(0,1-c*c)):.4f}")
# experts gate+up: IQ2_XXS in the file vs native 4-bit official
t,raw=gtensor(f"blk.{L}.ffn_gate_up_exps.weight"); E,R,C=reversed(t["dims"]); per=t["nbytes"]//E
n13=f"model.llm.layers.{L}.mlp.experts.w13_weight"; oraw=get_raw(n13); osc=get_raw(n13+"_scale"); operb=R*C//2; osper=R*(C//16)
print(f"\nexperts gate+up, layer {L}: file type {t['type']} ({8*t['nbytes']/(E*R*C):.2f} bits) vs official 4-bit")
coss=[]; rowc=[]; lay=None
for e in range(0,E,4):
    A=quants.dequantize(raw[e*per:(e+1)*per].reshape(R,-1),QT(t["type"])).reshape(R,C)
    B=decode_fp4(oraw[e*operb:(e+1)*operb], osc[e*osper:(e+1)*osper], R, C)
    if lay is None:  # find row layout: official interleaves gate/up rows, the file may de-interleave
        B2=np.concatenate([B[0::2],B[1::2]])
        c1=float((A*B).sum()/np.linalg.norm(A)/np.linalg.norm(B)); c2=float((A*B2).sum()/np.linalg.norm(A)/np.linalg.norm(B2)); lay="deinterleaved" if c2>c1 else "same"; print(f"  row layout in file: {lay} (cos same-order {c1:.3f}, de-interleaved {c2:.3f})")
    if lay=="deinterleaved": B=np.concatenate([B[0::2],B[1::2]])
    coss.append(float((A*B).sum()/np.linalg.norm(A)/np.linalg.norm(B)))
    rc=(A*B).sum(1)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)+1e-12); rowc.append(rc)
coss=np.array(coss); rowc=np.concatenate(rowc)
print(f"  per-expert cosine: mean {coss.mean():.4f} min {coss.min():.4f} max {coss.max():.4f}  -> relative error mean {np.sqrt(1-coss.mean()**2):.3f}")
print(f"  per-row cosine: mean {rowc.mean():.4f} p1 {np.percentile(rowc,1):.4f} p50 {np.percentile(rowc,50):.4f} p99 {np.percentile(rowc,99):.4f}")
print(f"  gate rows mean cos {rowc.reshape(-1,R)[:,:R//2].mean():.4f} | up rows mean cos {rowc.reshape(-1,R)[:,R//2:].mean():.4f}")
json.dump(dict(layer=L,expert_cos_mean=float(coss.mean()),expert_cos_min=float(coss.min()),row_cos_p1=float(np.percentile(rowc,1))),open(os.path.join(LAB,"results",f"damage_L{L}.json"),"w"))
