import numpy as np, s0b_blind as b
toy='rough'
U=b.random_starts(100000,toy); T=b.settle_time(U,toy); i=int(np.argmax(T))
traj=[U[i]]
for _ in range(T[i]): traj.append(b.F(traj[-1],toy))
lam,V,pr=b.spectrum_and_clvs(traj,toy)
print('T',T[i],'probe idx',pr,'lam',lam.round(3))
Vi=b.R.T@V  # internal components
np.set_printoptions(precision=3,suppress=True)
print('CLV internal |x1,x2,y| and |s|:'); print(np.vstack([np.abs(Vi[:3]),np.linalg.norm(Vi[3:],axis=0)]).T)
z=traj[pr]; print('z internal y',(z@b.R)[2],'remaining T',b.settle_time(z,toy))
ey=b.R[:,2]
for name,v in [('CLV3',V[:,2]),('true y',ey),('CLV1',V[:,0])]:
    for half in (0.05,0.2,0.8,3):
        W,TM=b.pim_line(z-half*v,z+half*v,toy,stop=1e-9)
        if np.ptp(TM)>0: break
    print(name,'half',half); print(' W',np.array2string(W[::2],precision=1)); print(' T',TM[::2])
