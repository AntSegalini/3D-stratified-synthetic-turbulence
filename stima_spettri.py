import numpy as np
import matplotlib.pyplot as plt

def stima_spettri(x,fsamp,nw):
    y=(x.T-np.mean(x,axis=-1).T).T
    q=0.5
    N=y.shape[-1]//nw
    uh=(8/3)**0.5*(1-(np.cos(np.pi*np.arange(N)/N))**2)
    kk=list(y.shape)
    kk[-1]=N//2+1
    powu=np.zeros(kk)
    WW=int(nw//q-1)
    for i in range(WW):
        powu+=np.abs(np.fft.rfft(y[...,np.arange(N)+int(i*N*q)]*uh))**2
    powu/=WW*N*fsamp/2 #the area is the variance	
    fr=np.arange(N//2+1)*fsamp/N
    return powu,fr

# fsamp=1000
# t=np.linspace(0,1,fsamp+1);t=t[:-1]
# x=np.sin(2*np.pi*t*100)
# print(np.var(x))
# powu,fr=stima_spettri(x,fsamp,4)
# print(np.trapz(powu,fr))
# plt.plot(fr,powu)
# plt.show()

