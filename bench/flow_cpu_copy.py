import numpy as np, sys
# GPU rule + no-climb clamp: an edge never carries more than 1/8 of its height difference (rounded at random),
# and nothing flows across a step of REPOSE or less. 4 edges x 1/8 can't lift a receiver above a giver.
N=int(sys.argv[2]); c=int(sys.argv[1]); H=np.zeros((N,N),np.int64); H[N//2-c//2:N//2+c//2, N//2-c//2:N//2+c//2]=c
FX=np.zeros((N,N)); FZ=np.zeros((N,N)); rng=np.random.default_rng(1)
REP,PUSH,FRIC,DAMP=1,0.5,0.5,0.98; total=H.sum()
def pe(H): return (H*(H+1)//2).sum()
def step(H,FX,FZ):
    out=[]
    for F,hn in ((FX,np.roll(H,-1,1)),(FZ,np.roll(H,-1,0))):
        d=H-hn
        fl=F*DAMP+PUSH*d; fl=np.sign(fl)*np.maximum(np.abs(fl)-FRIC,0)
        fl=np.where((np.abs(d)<=REP)&(np.abs(F)<FRIC),0.0,fl)
        fi=np.floor(fl+rng.random(H.shape)).astype(np.int64)
        lim=np.where(np.abs(d)>REP,np.floor(np.abs(d)/8+rng.random(H.shape)),0).astype(np.int64)
        fi=np.where(fi>0,np.minimum(fi,np.minimum(H//4,lim)),np.maximum(fi,-np.minimum(hn//4,lim)))
        out.append(fi)
    fx,fz=out; fx[:,-1]=0; fz[-1,:]=0
    return H-fx-fz+np.roll(fx,1,1)+np.roll(fz,1,0),fx.astype(float),fz.astype(float)
e0=pe(H); emax=e0; peakmax=H.max()
for r in range(1,int(sys.argv[3])+1):
    H,FX,FZ=step(H,FX,FZ); e=pe(H); emax=max(emax,e); peakmax=max(peakmax,H.max())
    if r%int(sys.argv[4])==0:
        row=H[N//2]; nz=np.nonzero(row)[0]
        print(f"  tick {r//4:4d}: peak {row.max():4d}  base width {nz[-1]-nz[0]+1:4d}  flowing {int((FX!=0).sum()+(FZ!=0).sum()):6d}")
print(f"cube {c}: conserved {H.sum()==total}; highest column ever {peakmax} (cube {c}); height energy ever rose: {emax>e0}")
