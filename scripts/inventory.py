import re, collections, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from official import header
H=header()["tensors"]; pat=collections.OrderedDict()
for k,v in H.items():
    if "visual" in k: continue
    key=re.sub(r"\.(\d+)\.",".N.",k); key=re.sub(r"\.(\d+)$",".K",key)
    e=pat.setdefault(key,[v["dtype"],v["shape"],0,0,set()]); e[2]+=1; e[3]+=v["data_offsets"][1]-v["data_offsets"][0]
    m=re.search(r"layers\.(\d+)\.",k)
    if m: e[4].add(int(m.group(1)))
for k,(dt,sh,n,sz,Ls) in pat.items():
    L=("L%d-%d(%d)"%(min(Ls),max(Ls),len(Ls))) if Ls else ""
    print(f"{k:62s} {dt:8s} {str(sh):22s} x{n:<4d} {sz/1e6:9.1f}MB {L}")
