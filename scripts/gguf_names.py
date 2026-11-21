import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggufmap import index
kv,T,mm=index(sys.argv[1])
print("tensors:",len(T)); print({k:v for k,v in kv.items() if v is not None and not k.startswith("tokenizer")})
for n,v in T.items():
    if n.startswith("blk.8.") or not n.startswith("blk."):
        e=1
        for d in v["dims"]: e*=d
        print(f"{n:38s} dims {str(v['dims']):22s} type {v['type']:3d} bytes {v['nbytes']:>11d} bits/elem {8*v['nbytes']/e:6.3f}")
