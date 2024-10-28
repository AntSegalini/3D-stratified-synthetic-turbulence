import numpy as np
from scipy.special import gamma
from scipy.io import loadmat

radice_pi=np.sqrt(np.pi)

################################################################################################################

def expressions_WKB_vectors(l,z):
    epsilon=1/np.sqrt(0j-l*(l+1))

    ww=np.arcsin(z)
    q=1-z**2

    P=q**(-1/4)*np.exp(ww/epsilon  -epsilon/8*(z/np.sqrt(q)+ww) +epsilon**2/(16*q))
    Q=q**(-1/4)*np.exp(-ww/epsilon +epsilon/8*(z/np.sqrt(q)+ww) +epsilon**2/(16*q))

    Pd=P*(1/(epsilon*np.sqrt(q))    +z/q/2  -epsilon/8*(2-z**2)/q**(3/2)    +epsilon**2*z/q**2/8)
    Qd=Q*(-1/(epsilon*np.sqrt(q))   +z/q/2  +epsilon/8*(2-z**2)/q**(3/2)    +epsilon**2*z/q**2/8)

    return P, Q, Pd, Qd

################################################################################################################

def legendre_PQ_WKB_vectors(l,z):
    epsilon=1/np.sqrt(0j-l*(l+1))
    F1,F2,F1d,F2d=expressions_WKB_vectors(l,z)

    ypp=1/epsilon-epsilon/4
    # expansion for P
    P0=radice_pi/(gamma((1-l)/2)*gamma(1+l/2))
    P0d=-2*radice_pi/(gamma((1+l)/2)*gamma(-l/2))
    cp=(P0+P0d/ypp)/2
    cm=(P0-P0d/ypp)/2
    P=cp*F1+cm*F2
    Pd=cp*F1d+cm*F2d

    # expansion for Q
    jj=gamma(1+l/2)/gamma((1+l)/2)
    K1=radice_pi*np.cos(np.pi*l/2)*jj
    K2=radice_pi*np.sin(np.pi*l/2)/(2*jj)

    P0=-K2
    P0d=K1
    cp=(P0+P0d/ypp)/2
    cm=(P0-P0d/ypp)/2
    Q=cp*F1+cm*F2
    Qd=cp*F1d+cm*F2d

    return P,Q, Pd, Qd

################################################################################################################

def legendre_PQ_0p(l,z):
    # series at z0=0+
    d1=gamma((1-l)/2)
    d2=gamma((1+l)/2)
    d3=gamma(1+l/2)

    # Gauss formulas (wikipedia)
    K1=radice_pi/(d1*d3)
    K2=-2*radice_pi/(d2*gamma(-l/2))

    a=[K1,K2]
    P=a[0]+a[1]*z
    Pd=a[1]*np.ones(z.shape,dtype=np.complex128)

    jj=d3/d2
    K1=radice_pi*np.cos(np.pi*l/2)*jj
    K2=radice_pi*np.sin(np.pi*l/2)/(2*jj)

    b=[-K2,K1]
    Q=b[0]+b[1]*z
    Qd=b[1]*np.ones(z.shape,dtype=np.complex128)

    for n in range(20):
        a.append(a[n]*(n**2+n-l*(1+l))/((n+1)*(n+2)))
        P+=a[n+2]*z**(n+2)
        Pd+=(n+2)*a[n+2]*z**(n+1)  
        b.append(b[n]*(n**2+n-l*(1+l))/((n+1)*(n+2)))
        Q+=b[n+2]*z**(n+2)
        Qd+=(n+2)*b[n+2]*z**(n+1)

    return P, Q, Pd, Qd

################################################################################################################

def legendre_PQ_z0(l,z,z0):
    # series at z0=+/-i;
    Phi=np.real(-l*(1+l))

    if np.imag(z0)>0:
        qqq=loadmat('hyperg_p1_v3.mat')
    else:
        qqq=loadmat('hyperg_m1_v3.mat')
    
    K1=np.interp(Phi,np.squeeze(qqq['Phi']),np.squeeze(qqq['P0']))
    K2=np.interp(Phi,np.squeeze(qqq['Phi']),np.squeeze(qqq['P0d']))
    K3=np.interp(Phi,np.squeeze(qqq['Phi']),np.squeeze(qqq['Q0']))
    K4=np.interp(Phi,np.squeeze(qqq['Phi']),np.squeeze(qqq['Q0d']))

    a=[K1,K2]
    P=a[0]+a[1]*(z-z0)
    Pd=a[1]*np.ones(z.shape,dtype=np.complex128)
    b=[K3,K4]
    Q=b[0]+b[1]*(z-z0)
    Qd=b[1]*np.ones(z.shape,dtype=np.complex128)

    for n in range(10):
        a.append((a[n+1]*2*z0*(n+1)**2 + a[n]*(n**2+n-l*(1+l)))/((1-z0**2)*(n+1)*(n+2)))
        P+=a[n+2]*(z-z0)**(n+2)
        Pd+=(n+2)*a[n+2]*(z-z0)**(n+1)
        b.append((b[n+1]*2*z0*(n+1)**2 + b[n]*(n**2+n-l*(1+l)))/((1-z0**2)*(n+1)*(n+2)))
        Q+=b[n+2]*(z-z0)**(n+2)
        Qd+=(n+2)*b[n+2]*(z-z0)**(n+1)

    return P, Q, Pd, Qd

################################################################################################################

def legendre_PQ_infinity(l,z):
    # series at infinity (l should be different than -1/2)
    Phi=-l*(1+l)

    # solution 1
    alpha1=-l
    P=np.ones(z.shape,dtype=np.complex128)
    Pd=-alpha1/z
    b=[1,0]

    # solution2
    alpha2=l+1
    Q=np.ones(z.shape,dtype=np.complex128)
    Qd=-alpha2/z
    c=[1,0]
    
    for n in range(0,20,2):
        b.append((n+alpha1)*(n+alpha1+1)/((n+alpha1+2)*(n+alpha1+1)+Phi)*b[n])
        b.append(0)
        P+=b[n+2]/z**(n+2)
        Pd-=b[n+2]*(n+2+alpha1)/z**(n+3)
        c.append((n+alpha2)*(n+alpha2+1)/((n+alpha2+2)*(n+alpha2+1)+Phi)*c[n])
        c.append(0)
        Q+=c[n+2]/z**(n+2)
        Qd-=c[n+2]*(n+2+alpha2)/z**(n+3)
    
    y1=P/z**alpha1
    y1d=Pd/z**alpha1
    y2=Q/z**alpha2
    y2d=Qd/z**alpha2

    #################################

    # coefficients are inverted compared to the JFM
    b2=gamma(-2*l-1)/gamma(-l)**2*2**(l+1)
    b1=gamma(2*l+1)/gamma(l+1)**2/2**l
    P=b1*y1+b2*y2
    Pd=b1*y1d+b2*y2d

    jj=gamma(1+l/2)/gamma((1+l)/2)
    K1=radice_pi*np.cos(np.pi*l/2)*jj
    K2=radice_pi*np.sin(np.pi*l/2)/(2*jj)

    # coefficients are inverted compared to the JFM
    pm=np.sign(np.imag(z))
    c2=(K1*1j**pm/2/gamma((1-l)/2)**2-K2/gamma(-l/2)**2)*radice_pi*gamma(-1/2-l)*1j**(pm*(1+l))
    c1=(K1*1j**pm/2/gamma(1+l/2)**2-K2/gamma((1+l)/2)**2)*radice_pi*gamma(1/2+l)*1j**(-pm*l)
    Q=c1*y1+c2*y2
    Qd=c1*y1d+c2*y2d

    return P, Q, Pd, Qd

################################################################################################################

def legendre_PQ_infinity_05(z):
    l=-1/2
    alpha1=-l
    P=np.ones(z.shape,dtype=np.complex128)
    Pd=-alpha1/z
    Q=np.zeros(z.shape,dtype=np.complex128)
    Qd=np.zeros(z.shape,dtype=np.complex128)
    b=[1,0]
    c=[0,0]
    
    for n in range(0,20,2):
        b.append((n+alpha1)*(n+alpha1+1)/((n+alpha1+2)*(n+alpha1+1)+1/4)*b[n])
        b.append(0)
        c.append((4*n+5)/(2*(n+2)**3)*b[n]+(2*n+1)*(2*n+3)/(4*(n+2)**2)*c[n])
        c.append(0)
        P+=b[n+2]/z**(n+2)
        Q+=c[n+2]/z**(n+2)
        Pd-=(n+2+1/2)*b[n+2]/z**(n+3)
        Qd-=(n+2+1/2)*c[n+2]/z**(n+3)
    
    y1=P/z**alpha1
    y1d=Pd/z**alpha1
    y2=(-np.log(z)*P+Q)/np.sqrt(z)
    y2d=(-P/z-np.log(z)*Pd+Qd)/np.sqrt(z)

    ##################################################################

    K=0.9360775742346218 # 3*np.sqrt(2)*np.log(2)/np.pi
    P=K*y1-np.sqrt(2)/np.pi*y2
    Pd=K*y1d-np.sqrt(2)/np.pi*y2d

    # jj=gamma(1+l/2)/gamma((1+l)/2)
    # K1=radice_pi*np.cos(np.pi*l/2)*jj
    # K2=radice_pi*np.sin(np.pi*l/2)/(2*jj)

    # K1=0.42360654239698947
    # K2=-1.854074677301372

    pm=np.sign(np.imag(z))

    # gg=0.577215664901532 #Euler Mascheroni constant
    # c1=K1/gamma(3/4)**2*1j**(pm*3/2)*(-pm*np.pi/2*1j-gg-digamma(3/4))-2*K2/gamma(1/4)**2*1j**(pm/2)*(-pm*np.pi/2*1j-gg-digamma(1/4))
    # c1*=radice_pi
    # c2=-(K1/gamma(3/4)**2*1j**(pm*3/2)-2*K2/gamma(1/4)**2*1j**(pm/2))
    # c2*=radice_pi
    
    c1=2.221441469079184+1.4703872152028215j*pm
    c2=-4.91954768840207e-17-0.7071067811865476j*pm
        
    Q=c1*y1+c2*y2
    Qd=c1*y1d+c2*y2d

    return P, Q, Pd, Qd

################################################################################################################

def legendre_PQ_l0(z):
    # series at l=0
    P=np.ones(z.shape,dtype=np.complex128)
    Pd=np.zeros(z.shape,dtype=np.complex128)
    Q=1/2*np.log((1+z)/(1-z))
    Qd=1/(1-z**2)

    return P, Q, Pd, Qd

################################################################################################################

def legendre_PQ_series_vectors(l,z):
    # this function is thought to handle z belonging to the imaginary axis
    # the Q function is taken from the formula of Wolfram

    Phi=np.real(-l*(1+l))

    P=np.zeros(z.shape,dtype=np.complex128)
    Q=np.zeros(z.shape,dtype=np.complex128)
    Pd=np.zeros(z.shape,dtype=np.complex128)
    Qd=np.zeros(z.shape,dtype=np.complex128)

    if np.abs(Phi)>6:
        P,Q, Pd, Qd=legendre_PQ_WKB_vectors(l,z)

    elif np.abs(Phi)==0:
        P,Q, Pd, Qd=legendre_PQ_l0(z)

    else:
        z1=np.where(np.abs(np.imag(z))<=0.5)[0]
        if len(z1)>0:
            P[z1],Q[z1], Pd[z1], Qd[z1]=legendre_PQ_0p(l,z[z1])
        
        z1=np.where((np.imag(z)>0.5) & (np.imag(z)<=1.6))[0]
        if len(z1)>0:
            P[z1],Q[z1], Pd[z1], Qd[z1]=legendre_PQ_z0(l,z[z1],1j)
        
        z1=np.where((np.imag(z)<-0.5) & (np.imag(z)>=-1.6))[0]
        if len(z1)>0:
            P[z1],Q[z1], Pd[z1], Qd[z1]=legendre_PQ_z0(l,z[z1],-1j)

        z1=np.where(np.abs(np.imag(z))>1.6)[0]
        if len(z1)>0 and Phi!=1/4:
            P[z1],Q[z1], Pd[z1], Qd[z1]=legendre_PQ_infinity(l,z[z1])
        elif len(z1)>0 and Phi==1/4:
            P[z1],Q[z1], Pd[z1], Qd[z1]=legendre_PQ_infinity_05(z[z1])

    return P, Q, Pd, Qd

#########################################################################################################


if __name__ == "__main__":
    # tests

    import matplotlib.pyplot as plt

    ##############################################################################
    
    def Legendre_equation(rho, z,lx):
        #here the variable z is 1j*t
        return [rho[1],(2*z*rho[1]-lx*(1+lx)*rho[0])/(1-z**2)]

    def Euler_integration(z,dz,IC,lx):
        out=[IC[0]]
        out_d=[IC[1]]
        for i in range(1,len(z)):
            c=Legendre_equation([out[-1],out_d[-1]], z[i],lx)
            out.append(out[-1]+dz*c[0])
            out_d.append(out_d[-1]+dz*c[1])
        return out,out_d

    def RK2_integration(z,dz,IC,lx):
        out=[IC[0]]
        out_d=[IC[1]]
        for i in range(1,len(z)):
            c=Legendre_equation([out[-1],out_d[-1]], z[i],lx)
            d=Legendre_equation([out[-1]+dz*c[0]/2,out_d[-1]+dz*c[1]/2], z[i]+dz/2,lx)
            out.append(out[-1]+dz*d[0])
            out_d.append(out_d[-1]+dz*d[1])
        return out,out_d

    def RK3_integration(z,dz,IC,lx):
        out=[IC[0]]
        out_d=[IC[1]]
        for i in range(1,len(z)):
            k1=Legendre_equation([out[-1],out_d[-1]], z[i],lx)
            k2=Legendre_equation([out[-1]+dz*k1[0]/2,out_d[-1]+dz*k1[1]/2], z[i]+dz/2,lx)
            k3=Legendre_equation([out[-1]+dz*(2*k2[0]-k1[0]),out_d[-1]+dz*(2*k2[1]-k1[1])], z[i]+dz,lx)
            out.append(out[-1]+dz/6*(k1[0]+4*k2[0]+k3[0]))
            out_d.append(out_d[-1]+dz/6*(k1[1]+4*k2[1]+k3[1]))
        return out,out_d

    def compare_stuff(P,Q,Pd,Qd,solP,solQ):
        plt.figure()
        plt.subplot(245)
        plt.plot(np.imag(z_int),np.real(solP[0]-P))
        plt.plot(np.imag(z_int),np.imag(solP[0]-P))
        plt.grid()
        plt.ylabel('error P')
        
        plt.subplot(246)
        plt.plot(np.imag(z_int),np.real(solP[1]-Pd))
        plt.plot(np.imag(z_int),np.imag(solP[1]-Pd))
        plt.grid()
        plt.ylabel('error Pd')

        plt.subplot(247)
        plt.plot(np.imag(z_int),np.real(solQ[0]-Q))
        plt.plot(np.imag(z_int),np.imag(solQ[0]-Q))
        plt.grid()
        plt.ylabel('error Q')
        
        plt.subplot(248)
        plt.plot(np.imag(z_int),np.real(solQ[1]-Qd))
        plt.plot(np.imag(z_int),np.imag(solQ[1]-Qd))
        plt.grid()
        plt.ylabel('error Qd')
 
        plt.subplot(241)
        plt.plot(np.imag(z_int),np.real(solP[0]))
        plt.plot(np.imag(z_int),np.imag(solP[0]))
        plt.plot(np.imag(z_int),np.real(P),'--k')
        plt.plot(np.imag(z_int),np.imag(P),'--r')
        plt.grid()
        plt.ylabel('P')
        
        plt.subplot(242)
        plt.plot(np.imag(z_int),np.real(solP[1]))
        plt.plot(np.imag(z_int),np.imag(solP[1]))
        plt.plot(np.imag(z_int),np.real(Pd),'--k')
        plt.plot(np.imag(z_int),np.imag(Pd),'--r')
        plt.grid()
        plt.ylabel('Pd')

        plt.subplot(243)
        plt.plot(np.imag(z_int),np.real(solQ[0]))
        plt.plot(np.imag(z_int),np.imag(solQ[0]))
        plt.plot(np.imag(z_int),np.real(Q),'--k')
        plt.plot(np.imag(z_int),np.imag(Q),'--r')
        plt.grid()
        plt.ylabel('Q')
        
        plt.subplot(244)
        plt.plot(np.imag(z_int),np.real(solQ[1]))
        plt.plot(np.imag(z_int),np.imag(solQ[1]))
        plt.plot(np.imag(z_int),np.real(Qd),'--k')
        plt.plot(np.imag(z_int),np.imag(Qd),'--r')
        plt.grid()
        plt.ylabel('Qd')

        print('error in P:',np.max(np.abs(solP[0]-P)),round(100*np.max(np.abs(solP[0]-P))/np.max(np.abs(P)),3),'%')
        print('error in Q:',np.max(np.abs(solQ[0]-Q)),round(100*np.max(np.abs(solQ[0]-Q))/np.max(np.abs(Q)),3),'%')
        print('error in Pd:',np.max(np.abs(solP[1]-Pd)),round(100*np.max(np.abs(solP[1]-Pd))/np.max(np.abs(Pd)),3),'%')
        print('error in Qd:',np.max(np.abs(solQ[1]-Qd)),round(100*np.max(np.abs(solQ[1]-Qd))/np.max(np.abs(Qd)),3),'%')
        plt.show()

    ##############################################################################

    ii=7 #select the case to test

    l=1
    
    ##############################################################################

    if ii==0:
        # test for legendre_PQ_0p
        z_int=np.linspace(0,0.5,20000)*1j
        P,Q,Pd,Qd=legendre_PQ_0p(l,z_int)

    ##############################################################################

    if ii==1:
        # test for legendre_PQ_z0 (+1j)
        z_int=np.linspace(0.5,1.6,20000)*1j
        P,Q,Pd,Qd=legendre_PQ_z0(l,z_int,1j)
    
    ##############################################################################

    if ii==2:
        # test for legendre_PQ_z0 (-1j)
        z_int=-np.linspace(0.5,1.6,20000)*1j
        P,Q,Pd,Qd=legendre_PQ_z0(l,z_int,-1j)

    ##############################################################################

    if ii==3:
        # test for legendre_PQ_infinity(l,z)
        z_int=-np.linspace(1.6,10,100000)*1j
        P,Q,Pd,Qd=legendre_PQ_infinity(l,z_int)

    ##############################################################################

    if ii==4:
        # test for legendre_PQ_infinity_05(z)
        l=-1/2 #(only)
        z_int=np.linspace(1.6,10,10000)*1j
        P,Q,Pd,Qd=legendre_PQ_infinity_05(z_int)

    ##############################################################################

    if ii==5:
        # test for legendre_PQ_WKB_vectors(l,z)
        z_int=np.linspace(0,10,1000000)*1j
        P,Q,Pd,Qd=legendre_PQ_WKB_vectors(l,z_int)

    if ii==6:
        z_int=np.linspace(0,10,200000)*1j
        P,Q,Pd,Qd=expressions_WKB_vectors(l,z_int)

    ##############################################################################

    if ii==7:
        z_int=np.linspace(-3.5,3.5,50000)*1j
        P,Q,Pd,Qd=legendre_PQ_series_vectors(l,z_int)

    ##############################################################################
    solP = RK3_integration(z_int,z_int[1]-z_int[0],[P[0],Pd[0]],l)
    solQ = RK3_integration(z_int,z_int[1]-z_int[0],[Q[0],Qd[0]],l)
    compare_stuff(P,Q,Pd,Qd,solP,solQ)