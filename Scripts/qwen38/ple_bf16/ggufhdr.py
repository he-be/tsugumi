import struct, sys, json
T={0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
def read(path, maxbytes=64<<20):
    b=open(path,'rb').read(maxbytes); o=[0]
    def u(fmt):
        v=struct.unpack_from(fmt,b,o[0]); o[0]+=struct.calcsize(fmt); return v[0]
    def s():
        n=u('<Q'); v=b[o[0]:o[0]+n]; o[0]+=n; return v.decode('utf-8','replace')
    def val(t):
        if t in T: return u(T[t])
        if t==8: return s()
        if t==9:
            et=u('<I'); n=u('<Q'); return [val(et) for _ in range(n)]
        raise ValueError(t)
    assert b[:4]==b'GGUF'; o[0]=4; ver=u('<I'); nt=u('<Q'); nkv=u('<Q')
    kv={}
    for _ in range(nkv):
        k=s(); t=u('<I'); kv[k]=val(t)
    align=kv.get('general.alignment',32)
    ts=[]
    for _ in range(nt):
        name=s(); nd=u('<I'); dims=[u('<Q') for _ in range(nd)]; ty=u('<I'); off=u('<Q')
        ts.append((name,dims,ty,off))
    start=(o[0]+align-1)//align*align
    return kv,ts,start
if __name__=='__main__':
    kv,ts,start=read(sys.argv[1])
    for k,v in kv.items():
        if isinstance(v,list) and len(v)>12: v=f'[{len(v)} items] {v[:6]}...'
        print('KV',k,v)
    print('data_start',start,'ntensors',len(ts))
    for t in ts: print('T',t[0],t[1],t[2],t[3])
