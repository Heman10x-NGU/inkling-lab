"""Minimal GGUF index: tensor name -> (dims, type id, absolute offset, nbytes). Tolerates unknown tensor types."""
import struct, mmap, os, numpy as np
def index(path):
    f=open(path,"rb"); mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ); p=0
    def rd(fmt):
        nonlocal p; v=struct.unpack_from(fmt,mm,p)[0]; p+=struct.calcsize(fmt); return v
    def rs():
        nonlocal p; n=rd("<Q"); s=bytes(mm[p:p+n]); p+=n; return s.decode("utf8","replace")
    sz={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}; fm={0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}
    def rv(t):
        nonlocal p
        if t==8: return rs()
        if t==9:
            et=rd("<I"); n=rd("<Q")
            if et in sz: p+=sz[et]*n; return None
            for _ in range(n): rv(et)
            return None
        return rd(fm[t])
    assert mm[:4]==b"GGUF"; p=4; rd("<I"); nt=rd("<Q"); nk=rd("<Q"); kv={}
    for _ in range(nk):
        k=rs(); t=rd("<I"); kv[k]=rv(t)
    infos=[]
    for _ in range(nt):
        name=rs(); nd=rd("<I"); dims=[rd("<Q") for _ in range(nd)]; t=rd("<I"); off=rd("<Q"); infos.append((name,dims,t,off))
    al=kv.get("general.alignment",32) or 32; start=(p+al-1)//al*al
    infos.sort(key=lambda x:x[3]); out={}; size=os.path.getsize(path)
    for i,(name,dims,t,off) in enumerate(infos):
        nxt=infos[i+1][3] if i+1<len(infos) else size-start
        out[name]=dict(dims=dims,type=t,off=start+off,nbytes=nxt-off)
    return kv,out,mm
