"""Partial reader for the official Inkling-2-Mini safetensors (HTTP range requests, cached as .npy)."""
import urllib.request, struct, json, os, numpy as np
U="https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/model-00001-of-00001.safetensors"
LAB=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE=os.path.join(LAB,"models","official_parts"); os.makedirs(CACHE,exist_ok=True)
def _rng(a,b):
    err=None
    for _ in range(4):
        try:
            r=urllib.request.Request(U,headers={"Range":f"bytes={a}-{b}"}); return urllib.request.urlopen(r,timeout=120).read()
        except Exception as e: err=e
    raise err
def header():
    p=os.path.join(CACHE,"header.json")
    if os.path.exists(p): return json.load(open(p))
    n=struct.unpack("<Q",_rng(0,7))[0]; h=json.loads(_rng(8,8+n-1)); h.pop("__metadata__",None)
    out={"base":8+n,"tensors":h}; json.dump(out,open(p,"w")); return out
def get(name):
    p=os.path.join(CACHE,name+".npy")
    if os.path.exists(p): return np.load(p)
    H=header(); v=H["tensors"][name]; a,b=v["data_offsets"]; raw=_rng(H["base"]+a,H["base"]+b-1)
    if v["dtype"]=="BF16": x=(np.frombuffer(raw,dtype="<u2").astype(np.uint32)<<16).view(np.float32)
    elif v["dtype"]=="F32": x=np.frombuffer(raw,dtype="<f4")
    elif v["dtype"]=="U8": x=np.frombuffer(raw,dtype=np.uint8)
    else: raise ValueError(v["dtype"])
    x=x.reshape(v["shape"]).copy(); np.save(p,x); return x

def get_raw(name):
    """Raw bytes of any tensor as uint8 (cached)."""
    p=os.path.join(CACHE,name+".raw.npy")
    if os.path.exists(p): return np.load(p,mmap_mode="r")
    H=header(); v=H["tensors"][name]; a,b=v["data_offsets"]; n=b-a; out=np.empty(n,dtype=np.uint8); step=64<<20
    for o in range(0,n,step):
        e=min(o+step,n); out[o:e]=np.frombuffer(_rng(H["base"]+a+o,H["base"]+a+e-1),dtype=np.uint8)
    np.save(p,out); return out
_E2M1=np.array([0,0.5,1,1.5,2,3,4,6,-0.0,-0.5,-1,-1.5,-2,-3,-4,-6],dtype=np.float32)
def _e4m3_table():
    t=np.zeros(256,dtype=np.float32)
    for b in range(256):
        s=-1.0 if b&0x80 else 1.0; e=(b>>3)&0xF; m=b&7
        t[b]=s*((m/8)*2.0**(-6) if e==0 else (1+m/8)*2.0**(e-7))
    return t
E4M3=_e4m3_table()
def decode_fp4(raw_bytes, scale_bytes, rows, cols):
    """One expert matrix [rows, cols] from packed E2M1 nibbles (2 per byte) and E4M3 per-16-element block scales."""
    b=np.asarray(raw_bytes); lo=_E2M1[b&0xF]; hi=_E2M1[b>>4]
    x=np.empty(b.size*2,dtype=np.float32); x[0::2]=lo; x[1::2]=hi; x=x.reshape(rows,cols//16,16)
    s=E4M3[np.asarray(scale_bytes)].reshape(rows,cols//16,1)
    return (x*s).reshape(rows,cols)
