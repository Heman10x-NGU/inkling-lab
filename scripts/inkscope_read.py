"""Reader for INKSCOPE_DUMP files: yields (name, ggml_type, shape, numpy array)."""
import struct, numpy as np
DT={0:np.float32,1:np.float16,24:np.int8,25:np.int16,26:np.int32,27:np.int64,28:np.float64}
def records(path):
    with open(path,"rb") as f:
        while True:
            h=f.read(64)
            if len(h)<64: return
            name=h.split(b"\0")[0].decode(); ty,n0,n1,n2,n3,nb=struct.unpack("<6q",f.read(48)); raw=f.read(nb)
            shape=[d for d in (n3,n2,n1,n0)]
            while len(shape)>1 and shape[0]==1: shape=shape[1:]
            cnt=n0*n1*n2*n3; a=np.frombuffer(raw,dtype=DT.get(ty,np.float32),count=cnt) if cnt*np.dtype(DT.get(ty,np.float32)).itemsize<=nb else None
            yield name,ty,shape,(a.reshape(shape) if a is not None else None)
