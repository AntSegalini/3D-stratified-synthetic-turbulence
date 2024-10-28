from legendre_functions import *

################################################################################################################

def isotropic_spectral(k1,k2,k3,Q_ratio): # Checked 20221210
    # k1 and k2 are single numbers, k3 is a vector

    PHI_ij=np.zeros((4,4,len(k3)))
    k0_2=k1**2+k2**2+k3**2
    
    PHI_ij[0,0,:]=k0_2-k1**2
    PHI_ij[0,1,:]=-k1*k2
    PHI_ij[0,2,:]=-k1*k3
    PHI_ij[1,0,:]=-k1*k2
    PHI_ij[1,1,:]=k0_2-k2**2
    PHI_ij[1,2,:]=-k2*k3
    PHI_ij[2,0,:]=-k1*k3
    PHI_ij[2,1,:]=-k2*k3
    PHI_ij[2,2,:]=k1**2+k2**2
    PHI_ij[:3,:3,:]*=(1+k0_2)**(-17/6)/(4*np.pi) # Karman Velocity spectrum normalized by 4*pi*k^4
    PHI_ij[3,3,:]=Q_ratio/(1+k0_2)**(11/6)
         
    return PHI_ij

################################################################################################################

def stratified_spectral(k1,k2,k3,N,alpha,Q_ratio,Gamma_value): # Checked 20221210
    # k1 and k2 are single numbers, k3 and t are vectors

    t=taux(alpha, Gamma_value, N,k1,k2,k3)
    k30=k3+alpha*k1*t
    
    kh=np.sqrt(k1**2+k2**2)
    k0=np.sqrt(kh**2+k30**2)

    A=np.eye(4)
    PHI_ij=isotropic_spectral(k1,k2,k30,Q_ratio)
    if k1!=0 and N!=0:
        z0=1j*k30/kh
        z=1j*k3/kh
        
        beta=-1j*alpha*k1/(N**2*kh)
        
        l=(-1+np.sqrt(1-4*(N*kh/(alpha*k1))**2+0j))/2 #lambda in Legendre functions

        P=np.empty(k3.shape,dtype=np.complex128)
        Pd=np.empty(k3.shape,dtype=np.complex128)
        Q=np.empty(k3.shape,dtype=np.complex128)
        Qd=np.empty(k3.shape,dtype=np.complex128)

        P0=np.empty(k30.shape,dtype=np.complex128)
        Pd0=np.empty(k30.shape,dtype=np.complex128)
        Q0=np.empty(k30.shape,dtype=np.complex128)
        Qd0=np.empty(k30.shape,dtype=np.complex128)

        W=np.empty(k30.shape,dtype=np.complex128)

        if np.abs((N*kh/(alpha*k1))**2)<6:
            P0, Q0, Pd0, Qd0=legendre_PQ_series_vectors(l,z0)
            P, Q, Pd, Qd=legendre_PQ_series_vectors(l,z)
            W=1-z0**2
        
        else:
            P0, Q0, Pd0, Qd0=expressions_WKB_vectors(l,z0)
            P, Q, Pd, Qd=expressions_WKB_vectors(l,z)
            W=1/(P0*Qd0-Pd0*Q0)
        
        # Equations 2.25 and 2.26
        F_Arho0=W*Qd0
        F_Brho0=-W*Pd0
        F_Aw0=-W*Q0/beta
        F_Bw0=W*P0/beta

        # Equations 2.29-2.32
        F_ww0=  beta*(F_Aw0*Pd+F_Bw0*Qd)
        F_wrho0=beta*(F_Arho0*Pd+F_Brho0*Qd)
        F_rhow0=    F_Aw0*P+F_Bw0*Q
        F_rhorho0=F_Arho0*P+F_Brho0*Q

        # Equations 2.19-2.20
        F_uA=alpha/(N*kh)**2*(k1**2*(z*Pd-z0*Pd0)-k2**2*(P-P0))
        F_uB=alpha/(N*kh)**2*(k1**2*(z*Qd-z0*Qd0)-k2**2*(Q-Q0))

        # Equations 2.21-2.22
        F_vA=alpha*k1*k2/(N*kh)**2*(z*Pd-z0*Pd0+P-P0)
        F_vB=alpha*k1*k2/(N*kh)**2*(z*Qd-z0*Qd0+Q-Q0)

        # Equations 2.28...
        F_urho0=F_uA*F_Arho0    +   F_uB*F_Brho0
        F_uw0=  F_uA*F_Aw0      +   F_uB*F_Bw0    
        F_vrho0=F_vA*F_Arho0    +   F_vB*F_Brho0
        F_vw0=  F_vA*F_Aw0      +   F_vB*F_Bw0    
        
        for i in range(len(k3)):
            A[:,2:]=  np.real(np.array([[  F_uw0[i],    F_urho0[i]],
                                        [  F_vw0[i],    F_vrho0[i]], 
                                        [  F_ww0[i],    F_wrho0[i]], 
                                        [  F_rhow0[i],  F_rhorho0[i]]]))

            PHI_ij[:,:,i]=np.matmul(A,np.matmul(PHI_ij[:,:,i],A.T))

    elif k1==0 and k2!=0 and N!=0:
        Phi=np.sqrt(N**2+0j)*abs(k2)/k0
        S=np.sin(Phi*t)
        C=np.cos(Phi*t)
        for i in range(len(k3)):            

            A[:,2:]=  np.real(np.array([[   -alpha*S[i]/Phi[i],      -alpha/N**2*(C[i]-1)], 
                                        [    -k30[i]/k2*(C[i]-1),     Phi[i]*k30[i]/(k2*N**2)*S[i]], 
                                        [   C[i],                    -Phi[i]/N**2*S[i]], 
                                        [   N**2/Phi[i]*S[i],        C[i]]]))
            
            PHI_ij[:,:,i]=np.matmul(A,np.matmul(PHI_ij[:,:,i],A.T))
    
    elif k1==0 and k2==0 and N!=0:
        for i in range(len(k3)):            

            A[:,2:]=  np.real(np.array([[  -alpha*t[i],      0], 
                                        [  0,                0], 
                                        [  1,                0], 
                                        [  N**2*t[i],        1]]))

            PHI_ij[:,:,i]=np.matmul(A,np.matmul(PHI_ij[:,:,i],A.T))

    elif k1!=0 and N==0:
        # Mann 94 formulation 
        for i in range(len(k3)):
            p1=(np.arctan(k3[i]/kh)-np.arctan(k30[i]/kh))/kh
            p2=k3[i]/(kh**2+k3[i]**2)-k30[i]/k0[i]**2
            A[:,2:]=  np.real(np.array([[    k0[i]**2*k2**2/(k1*kh**2)*p1 -k0[i]**2*k1/kh**2*p2,    0],
                                        [    -k0[i]**2*k2/kh**2*(p1+p2),    0], 
                                        [   (kh**2+k30[i]**2)/(kh**2+k3[i]**2),    0], 
                                        [    0,  1]]))

            PHI_ij[:,:,i]=np.matmul(A,np.matmul(PHI_ij[:,:,i],A.T))

    elif k1==0 and N==0:
        # Mann 94 formulation 
        for i in range(len(k3)):
            A[:,2:]=  np.real(np.array([[    -alpha*t[i],    0],
                                        [    0,    0], 
                                        [   1,    0], 
                                        [    0,  1]]))
            
            PHI_ij[:,:,i]=np.matmul(A,np.matmul(PHI_ij[:,:,i],A.T))

    return PHI_ij

################################################################################################################

def taux(alpha, Gamma, N,k1,k2,k3): # Checked 20221209

    kLx=np.sqrt(k1**2+k2**2+k3**2)

    tau_Mann=np.empty(kLx.shape)
    
    tau_Mann[kLx<=1e-5]=1.2053*1e6 #saturated time scale
    
    tau_Mann[(kLx<=0.05) & (kLx>1e-5)]=1.2053/kLx[(kLx<=0.05) & (kLx>1e-5)]
    
    tau_Mann[kLx>=20]=kLx[kLx>=20]**(-2/3)

    z=np.where((kLx>0.05) & (kLx<20))[0]
    if len(z)>0:
        kLu=np.array([  [0.0500,  24.1061],
                        [0.0531,   22.6905],
                        [0.0564,   21.3580],
                        [0.0600,   20.1037],
                        [0.0637,   18.9232],
                        [0.0677,   17.8119],
                        [0.0719,   16.7659],
                        [0.0764,   15.7813],
                        [0.0811,   14.8546],
                        [0.0862,   13.9822],
                        [0.0916,   13.1611],
                        [0.0973,   12.3882],
                        [0.1034,   11.6608],
                        [0.1098,   10.9760],
                        [0.1167,   10.3314],
                        [0.1239,    9.7247],
                        [0.1317,    9.1536],
                        [0.1399,    8.6161],
                        [0.1486,    8.1101],
                        [0.1579,    7.6339],
                        [0.1677,    7.1856],
                        [0.1782,    6.7637],
                        [0.1893,    6.3665],
                        [0.2011,    5.9927],
                        [0.2137,    5.6408],
                        [0.2270,    5.3096],
                        [0.2412,    4.9979],
                        [0.2562,    4.7045],
                        [0.2722,    4.4284],
                        [0.2892,    4.1685],
                        [0.3072,    3.9239],
                        [0.3264,    3.6938],
                        [0.3468,    3.4772],
                        [0.3684,    3.2734],
                        [0.3914,    3.0816],
                        [0.4158,    2.9013],
                        [0.4417,    2.7316],
                        [0.4693,    2.5721],
                        [0.4986,    2.4221],
                        [0.5297,    2.2811],
                        [0.5627,    2.1486],
                        [0.5978,    2.0241],
                        [0.6351,    1.9073],
                        [0.6748,    1.7976],
                        [0.7169,    1.6947],
                        [0.7616,    1.5983],
                        [0.8091,    1.5079],
                        [0.8596,    1.4232],
                        [0.9132,    1.3439],
                        [0.9702,    1.2697],
                        [1.0307,    1.2003],
                        [1.0950,    1.1354],
                        [1.1633,    1.0747],
                        [1.2359,    1.0180],
                        [1.3130,    0.9650],
                        [1.3950,    0.9154],
                        [1.4820,    0.8690],
                        [1.5744,    0.8256],
                        [1.6727,    0.7849],
                        [1.7770,    0.7468],
                        [1.8879,    0.7111],
                        [2.0057,    0.6775],
                        [2.1308,    0.6459],
                        [2.2637,    0.6162],
                        [2.4050,    0.5882],
                        [2.5550,    0.5618],
                        [2.7144,    0.5369],
                        [2.8838,    0.5133],
                        [3.0637,    0.4910],
                        [3.2548,    0.4698],
                        [3.4579,    0.4497],
                        [3.6736,    0.4306],
                        [3.9028,    0.4125],
                        [4.1463,    0.3952],
                        [4.4050,    0.3787],
                        [4.6798,    0.3631],
                        [4.9718,    0.3481],
                        [5.2820,    0.3338],
                        [5.6115,    0.3202],
                        [5.9616,    0.3071],
                        [6.3335,    0.2947],
                        [6.7287,    0.2827],
                        [7.1484,    0.2713],
                        [7.5944,    0.2604],
                        [8.0682,    0.2499],
                        [8.5716,    0.2399],
                        [9.1064,    0.2303],
                        [9.6745,    0.2211],
                        [10.2781,    0.2122],
                        [10.9193,    0.2038],
                        [11.6006,    0.1957],
                        [12.3243,    0.1879],
                        [13.0932,    0.1804],
                        [13.9101,    0.1732],
                        [14.7779,    0.1663],
                        [15.6999,    0.1597],
                        [16.6794,    0.1534],
                        [17.7200,    0.1473],
                        [18.8255,    0.1414],
                        [20.0000,    0.1358]])
        tau_Mann[z]=np.interp(kLx[z],kLu[:,0],kLu[:,1])
    
    tau_Mann*=Gamma/alpha

    if np.real(N**2)>0:
        return tau_Mann/(0.15*N*tau_Mann+1)
    else:
        return tau_Mann # for the unstable case the tau_Mann is returned
        