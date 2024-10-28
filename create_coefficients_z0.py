import numpy as np
from scipy.special import hyp2f1
from scipy.special import gamma
from scipy.io import loadmat,savemat

radice_pi=np.sqrt(np.pi)
z0=1j
Phi=np.linspace(-6,6,100)
l=(-1+np.sqrt(1+0j-4*Phi))/2

P0=np.zeros(Phi.shape,dtype=np.complex128)
P0d=np.zeros(Phi.shape,dtype=np.complex128)
c1=np.zeros(Phi.shape,dtype=np.complex128)
c2=np.zeros(Phi.shape,dtype=np.complex128)
c3=np.zeros(Phi.shape,dtype=np.complex128)
c4=np.zeros(Phi.shape,dtype=np.complex128)

for i in range(len(Phi)):
    P0[i]=hyp2f1(-l[i],l[i]+1,1,(1-z0)/2)
    P0d[i]=-Phi[i]/2*hyp2f1(1-l[i],l[i]+2,2,(1-z0)/2)
    c1[i]=hyp2f1((1-l[i])/2,1+l[i]/2,3/2,z0**2)
    c2[i]=hyp2f1(-l[i]/2,(1+l[i])/2,1/2,z0**2)
    c3[i]=hyp2f1((3-l[i])/2,2+l[i]/2,5/2,z0**2)
    c4[i]=hyp2f1(1-l[i]/2,(3+l[i])/2,3/2,z0**2)

d2=gamma((1+l)/2)
d3=gamma(1+l/2)
jj=d3/d2
K1=radice_pi*np.cos(np.pi*l/2)*jj
K2=radice_pi*np.sin(np.pi*l/2)/(2*jj)

Q0=z0*K1*c1-K2*c2
Q0d=K1*(c1+z0**2*(1-l)*(2+l)/3*c3)+K2*z0*l*(1+l)*c4