from create_spectrum import *
import multiprocessing as mpr
from concurrent.futures import ProcessPoolExecutor

################################################################################################################

def caller_stratified_spectral(lista):
    return stratified_spectral(lista['k1'],lista['k2'],lista['k3'],lista['N'],lista['alpha'],lista['Q_ratio'],lista['Gamma_value']),lista['i'],lista['j']

################################################################################################################

def caller_isotropic_spectral(lista):
    return isotropic_spectral(lista['k1'],lista['k2'],lista['k3'],lista['Q_ratio']),lista['i'],lista['j']

################################################################################################################

def get_wavenumbers(Nx,Ny,Nz,Lx,Ly,Lz): # Checked 20221210
    """
    Inputs:
    Nx= discretisation in the X direction
    Ny= discretisation in the Y direction
    Nz= discretisation in the Z direction (This can be large without slowing down the code)
    Lx= Domain length in the X direction ( x=np.linspace(0,Lx*(1-1/Nx),Nx) )
    Ly= Domain length in the Y direction
    Lz= Domain length in the Z direction
    
    Outputs:
    k1= X wavenumber [1/m]
    k2= Y wavenumber [1/m]
    k3= Z wavenumber [1/m]
    """

    # the size of the domain is NOT Lx but rather Lx*(1-1/Nx) since Lx imposes the periodicity
    k1= np.fft.rfftfreq(Nx,Lx/Nx/(2*np.pi)) 
    k2=  np.fft.fftfreq(Ny,Ly/Ny/(2*np.pi)) 
    k3=  np.fft.fftfreq(Nz,Lz/Nz/(2*np.pi)) 
    return k1,k2,k3

################################################################################################################

def Compute_PHI(dU_dz,Brunt_Vaisala_Frequency,Us,L,k1,k2,k3): # Checked 20221210
    """
    Routine to create spectrum for synthetic turbulence

    Inputs:
    dU_dz= velocity shear [1/s] (if that is zero than the isotropic spectrum is used)
    Brunt_Vaisala_Frequency= Brunt Väisälä frequency [1/s]
    Us= streamwise velocity standard deviation [m/s] (used as characteristic velocity scale)
    L= Characteristic length scale [m]
    k1= wavenumber in the X direction
    k2= wavenumber in the Y direction
    k3= wavenumber in the Z direction
    
    Outputs:
    PHI= Spectral tensor (According to Segalini & Arnqvist, 2015, JFM)
    """
    # Us is given by the desired u'

    Gamma_value=3.78

    alpha=L/Us*dU_dz
    N=L/Us*Brunt_Vaisala_Frequency
    
    Ri=0
    if dU_dz!=0:
        Ri=(N/alpha)**2
        
    Q_ratio=0.712*Ri**2+0.04*Ri

    PHI=np.zeros((4,4,len(k3),len(k2),len(k1)))

    lista=[]
    for i in range(len(k1)):
        for j in range(len(k2)):
            lista.append({'i':i,'j':j,'k1':k1[i]*L,'k2':k2[j]*L,'k3':k3*L,'N':N,'alpha':alpha,'Q_ratio':Q_ratio,'Gamma_value':Gamma_value})

    with ProcessPoolExecutor(max_workers=min([mpr.cpu_count(),30])) as executor:
        
        if dU_dz==0:
            results = executor.map(caller_isotropic_spectral, lista)
        else:
            results = executor.map(caller_stratified_spectral, lista)

        for result in results:
            PHI[:,:,:,result[2],result[1]]=result[0]

    return PHI

#######################################################################################

def Synthetic_turbulence(dU_dz,Brunt_Vaisala_Frequency,Us,T0,L,Nx,Ny,Nz,Lx,Ly,Lz,aliasing_correction=0,periodicity_correction=False): # Checked 20221208
    """
    Routine to create synthetic turbulence

    Inputs:
    dU_dz= velocity shear [1/s] (if that is zero than the isotropic spectrum is used)
    Brunt_Vaisala_Frequency= Brunt Väisälä frequency [1/s]
    Us= streamwise velocity standard deviation [m/s] (used as characteristic velocity scale)
    T0= Reference temperature [K]
    L= Characteristic length scale [m]
    Nx= discretisation in the X direction
    Ny= discretisation in the Y direction
    Nz= discretisation in the Z direction (This can be large without slowing down the code)
    Lx= Domain length in the X direction ( x=np.linspace(0,Lx*(1-1/Nx),Nx) )
    Ly= Domain length in the Y direction ( same as above )
    Lz= Domain length in the Z direction ( same as above )
    aliasing_correction=(0,1,2,3) = This is the level used to correct for aliasing 
    (0 implies no correction, 1 implies one level around ky and kz, 2 implies two levels around ky and kz, 3 implies three levels around ky and kz)

    Outputs:
    U= X velocity fluctuation [m/s] (order is [Z,Y,X])
    V= Y velocity fluctuation [m/s]
    W= Z velocity fluctuation [m/s]
    T= temperature fluctuation [K]
    
    """
    
    k1,k2,k3=get_wavenumbers(Nx,Ny,Nz,Lx,Ly,Lz)

    # compensate_aliasing
    if aliasing_correction==1:
        # 1st level
        k3_bis=np.concatenate([ k3,k3+2*np.pi*Nz/Lz,k3-2*np.pi*Nz/Lz])
        PHI = Compute_PHI(dU_dz,Brunt_Vaisala_Frequency,Us,L,k1,k2,k3_bis)
        for ii in [-1,1]:
            PHI += Compute_PHI(dU_dz,Brunt_Vaisala_Frequency,Us,L,k1,k2+ii*2*np.pi*Ny/Ly,k3_bis)
        PHI=PHI[:,:,:Nz]+PHI[:,:,Nz:2*Nz]+PHI[:,:,2*Nz:]
        
    elif aliasing_correction==2:
        # 2nd level
        k3_bis=np.concatenate([ k3,k3+2*np.pi*Nz/Lz,k3-2*np.pi*Nz/Lz,\
                                k3+4*np.pi*Nz/Lz,k3-4*np.pi*Nz/Lz])
        PHI = Compute_PHI(dU_dz,Brunt_Vaisala_Frequency,Us,L,k1,k2,k3_bis)
        for ii in [-2,-1,1,2]:
            PHI += Compute_PHI(dU_dz,Brunt_Vaisala_Frequency,Us,L,k1,k2+ii*2*np.pi*Ny/Ly,k3_bis)
        PHI=PHI[:,:,:Nz] +  PHI[:,:,Nz:2*Nz]+PHI[:,:,2*Nz:3*Nz]+\
                            PHI[:,:,3*Nz:4*Nz]+PHI[:,:,4*Nz:]

    elif aliasing_correction==3:
        # 3rd level
        k3_bis=np.concatenate([ k3,k3+2*np.pi*Nz/Lz,k3-2*np.pi*Nz/Lz,\
                                k3+4*np.pi*Nz/Lz,k3-4*np.pi*Nz/Lz,\
                                k3+6*np.pi*Nz/Lz,k3-6*np.pi*Nz/Lz])
        PHI = Compute_PHI(dU_dz,Brunt_Vaisala_Frequency,Us,L,k1,k2,k3_bis)
        for ii in [-3,-2,-1,1,2,3]:
            PHI += Compute_PHI(dU_dz,Brunt_Vaisala_Frequency,Us,L,k1,k2+ii*2*np.pi*Ny/Ly,k3_bis)
        PHI=PHI[:,:,:Nz]+PHI[:,:,Nz:2*Nz]+PHI[:,:,2*Nz:3*Nz]+ \
            PHI[:,:,3*Nz:4*Nz]+PHI[:,:,4*Nz:5*Nz]+\
            PHI[:,:,5*Nz:6*Nz]+PHI[:,:,6*Nz:]           
    else:
        # no aliasing
        PHI=Compute_PHI(dU_dz,Brunt_Vaisala_Frequency,Us,L,k1,k2,k3)
        
    ############################### Discretisation Fix ###############################
    
    if periodicity_correction:
        from scipy.interpolate import interp2d
        q1=np.where(k1<3/L)[0]
        q2=np.where(np.abs(k2)<3/L)[0]
        q3=np.where(np.abs(k3)<3/L)[0]
        ky=np.linspace(-1,1,20)*2*np.pi/Ly #integration to the first zero of the sinc
        kz=np.linspace(-1,1,21)*2*np.pi/Lz #integration to the first zero of the sinc
        Er=(np.repeat((np.sinc(ky*Ly/(2*np.pi))**2)[:,np.newaxis],len(kz),axis=-1)*np.sinc(kz*Lz/(2*np.pi))**2).T
        factor=(ky[1]-ky[0])*(kz[1]-kz[0])*Ly*Lz/(2*np.pi*0.902853)**2 # This has been calibrated to give the integral of unitari PHI equal to 1
        M=np.zeros((4,4)) 
        change=[]            
        for i in q1:
            for j in q2:
                for k in q3:
                    if k1[i]**2+k2[j]**2+k3[k]**2<9/L**2:
                        for ii in range(4):
                            for jj in range(4):
                                S=interp2d(np.fft.fftshift(k2),np.fft.fftshift(k3),np.fft.fftshift(PHI[ii,jj,:,:,i],axes=(0,-1)))
                                M[ii,jj]=np.trapz( np.trapz(S(k2[j]+ky,k3[k]+kz)*Er)) 
                        change.append([k,j,i,M * factor])
        for i in range(len(change)):
            PHI[:,:,change[i][0],change[i][1],change[i][2]]=change[i][3]
        change=[]
    
    ############################################################################

    A=np.zeros((Nz*Ny*(Nx//2+1),4,4),dtype=np.complex128)

    A[:,0,0]=np.sqrt(np.abs(np.reshape(PHI[0,0,:,:,:],Nz*Ny*(Nx//2+1))))
    A[:,1,0]=np.reshape(PHI[0,1,:,:,:],Nz*Ny*(Nx//2+1))
    A[:,2,0]=np.reshape(PHI[0,2,:,:,:],Nz*Ny*(Nx//2+1))
    A[:,3,0]=np.reshape(PHI[0,3,:,:,:],Nz*Ny*(Nx//2+1))
    zz=np.where(A[:,0,0]!=0)[0]
    A[zz,1,0]/=A[zz,0,0]
    A[zz,2,0]/=A[zz,0,0]
    A[zz,3,0]/=A[zz,0,0]
    zz=np.where(A[:,0,0]==0)[0]
    A[zz,1,0]=0
    A[zz,2,0]=0
    A[zz,3,0]=0
    
    A[:,1,1]=np.sqrt(np.reshape(PHI[1,1,:,:,:],Nz*Ny*(Nx//2+1))-np.abs(A[:,1,0])**2+0j)
    A[:,2,1]=np.reshape(PHI[1,2,:,:,:],Nz*Ny*(Nx//2+1))-np.conj(A[:,1,0])*A[:,2,0]
    A[:,3,1]=np.reshape(PHI[1,3,:,:,:],Nz*Ny*(Nx//2+1))-np.conj(A[:,1,0])*A[:,3,0]
    zz=np.where(A[:,1,1]!=0)[0]
    A[zz,2,1]/=np.conj(A[zz,1,1])
    A[zz,3,1]/=np.conj(A[zz,1,1])
    zz=np.where(A[:,1,1]==0)[0]
    A[zz,2,1]=0
    A[zz,3,1]=0
    
    A[:,2,2]=np.sqrt(np.reshape(PHI[2,2,:,:,:],Nz*Ny*(Nx//2+1))-np.abs(A[:,2,0])**2-np.abs(A[:,2,1])**2+0j)
    A[:,3,2]=np.reshape(PHI[2,3,:,:,:],Nz*Ny*(Nx//2+1))-np.conj(A[:,2,0])*A[:,3,0]-np.conj(A[:,2,1])*A[:,3,1]
    zz=np.where(A[:,2,2]!=0)[0]
    A[zz,3,2]/=np.conj(A[zz,2,2])
    zz=np.where(A[:,2,2]==0)[0]
    A[zz,3,2]=0
    
    A[:,3,3]=np.sqrt(np.reshape(PHI[3,3,:,:,:],Nz*Ny*(Nx//2+1))-np.abs(A[:,3,0])**2-np.abs(A[:,3,1])**2-np.abs(A[:,3,2])**2+0j)

    t= (np.random.randn(4,Nz*Ny*(Nx//2+1))+1j*np.random.randn(4,Nz*Ny*(Nx//2+1)))/np.sqrt(2) 
    
    # Nb=1;    print(np.round(np.conj(A[Nb,:,:]@(A[Nb,:,:].T))-np.reshape(PHI,(4,4,Nz*Ny*(Nx//2+1)))[:,:,Nb],6))

    t[3,:]=A[:,3,0]*t[0,:]+A[:,3,1]*t[1,:]+A[:,3,2]*t[2,:]+A[:,3,3]*t[3,:]
    t[2,:]=A[:,2,0]*t[0,:]+A[:,2,1]*t[1,:]+A[:,2,2]*t[2,:]
    t[1,:]=A[:,1,0]*t[0,:]+A[:,1,1]*t[1,:]
    t[0,:]*=A[:,0,0]
    
    ##########################################################################

    U=np.fft.irfftn(np.reshape(t[0,:],(Nz,Ny,Nx//2+1)),axes=[0,1,2]);U-=np.mean(U)
    factor=Us/np.std(U);    U*=factor

    V=np.fft.irfftn(np.reshape(t[1,:],(Nz,Ny,Nx//2+1)),axes=[0,1,2]);V-=np.mean(V); V*=factor
    W=np.fft.irfftn(np.reshape(t[2,:],(Nz,Ny,Nx//2+1)),axes=[0,1,2]);W-=np.mean(W); W*=factor
    # minus sign spotted by Johan
    T=-np.fft.irfftn(np.reshape(t[3,:],(Nz,Ny,Nx//2+1)),axes=[0,1,2])*T0*Us/(9.81*L);T-=np.mean(T); T*=factor
    
    return U,V,W,T
    
################################################################################################################

if __name__ == "__main__":
    ## tests
    import matplotlib.pyplot as plt
    import time 

    ###################################################################

    def stima_spettri(x,fsamp,nw):
        y=(x.T-np.mean(x,axis=-1).T).T
        q=0.5
        N=y.shape[-1]//nw
        uh=(8/3)**0.5*(1-np.cos(np.pi*np.arange(N)/N)**2)
        kk=list(y.shape)
        kk[-1]=N//2+1
        powu=np.zeros(kk)
        WW=int(nw/q-1)
        for i in range(WW):
            powu+=np.abs(np.fft.rfft(y[...,np.arange(N)+int(i*N*q)]*uh))**2
        powu/=WW*N*fsamp/2 #the area is the variance	
        fr=np.arange(N//2+1)*fsamp/N
        return powu,fr

    ###################################################################

    def stima_crossspettri(x1,x2,fsamp,nw):
        y1=(x1.T-np.mean(x1,axis=-1).T).T
        y2=(x2.T-np.mean(x2,axis=-1).T).T
        q=0.5
        N=y1.shape[-1]//nw
        uh=(8/3)**0.5*(1-(np.cos(np.pi*np.arange(N)/N))**2)
        kk=list(y1.shape)
        kk[-1]=N//2+1
        powu=np.zeros(kk,dtype=np.complex128)
        WW=int(nw//q-1)
        for i in range(WW):
            powu+=np.conj(np.fft.rfft(y1[...,np.arange(N)+int(i*N*q)]*uh))*np.fft.rfft(y2[...,np.arange(N)+int(i*N*q)]*uh)
        powu/=WW*N*fsamp/2 #the area is the variance	
        fr=np.arange(N//2+1)*fsamp/N
        return powu,fr

    ###################################################################

    def get_spectrum_from_PHI(k1,PHI):
        Nkx=len(k1)
        Pxx=np.zeros(Nx//2+1)
        Pxy=np.zeros(Pxx.shape,dtype=np.complex128)
        Pxz=np.zeros(Pxx.shape,dtype=np.complex128)
        Pxt=np.zeros(Pxx.shape,dtype=np.complex128)
        Pyy=np.zeros(Pxx.shape)
        Pyz=np.zeros(Pxx.shape,dtype=np.complex128)
        Pyt=np.zeros(Pxx.shape,dtype=np.complex128)
        Pzz=np.zeros(Pxx.shape)
        Pzt=np.zeros(Pxx.shape,dtype=np.complex128)
        Ptt=np.zeros(Pxx.shape)
        for i in range(Nkx):
            Pxx[i]=np.real(np.sum(PHI[0,0,:,:,i]))
            Pxy[i]=np.sum(PHI[0,1,:,:,i])
            Pxz[i]=np.sum(PHI[0,2,:,:,i])
            Pxt[i]=np.sum(PHI[0,3,:,:,i])
            Pyy[i]=np.real(np.sum(PHI[1,1,:,:,i]))
            Pyz[i]=np.sum(PHI[1,2,:,:,i])
            Pyt[i]=np.sum(PHI[1,3,:,:,i])
            Pzz[i]=np.real(np.sum(PHI[2,2,:,:,i]))
            Pzt[i]=np.sum(PHI[2,3,:,:,i])
            Ptt[i]=np.real(np.sum(PHI[3,3,:,:,i]))

        s=np.trapz(Pxx,k1)
        Pxx/=s
        Pxy/=s
        Pxz/=s
        Pxt/=s
        Pyy/=s
        Pyz/=s
        Pyt/=s
        Pzz/=s
        Pzt/=s
        Ptt/=s
        return Pxx,Pxy,Pxz,Pxt,Pyy,Pyz,Pyt,Pzz,Pzt,Ptt

    ###################################################################

    def get_spectra_from_time_series(U,V,W,T,Nwindows=2):
        Nz,Ny,Nx=U.shape
        
        pow_uu,fr=stima_spettri(U[0,0,:],1/(Lx/Nx),Nwindows)
        pow_uu*=0
        pow_uv=np.zeros(pow_uu.shape,dtype=np.complex128)
        pow_uw=np.zeros(pow_uu.shape,dtype=np.complex128)
        pow_ut=np.zeros(pow_uu.shape,dtype=np.complex128)
        pow_vv=np.zeros(pow_uu.shape)
        pow_vw=np.zeros(pow_uu.shape,dtype=np.complex128)
        pow_vt=np.zeros(pow_uu.shape,dtype=np.complex128)
        pow_ww=np.zeros(pow_uu.shape)
        pow_wt=np.zeros(pow_uu.shape,dtype=np.complex128)
        pow_tt=np.zeros(pow_uu.shape)

        P=np.zeros(pow_uu.shape)
        Pc=np.zeros(pow_uu.shape,dtype=np.complex128)

        for i in range(Ny):
            for j in range(Nz):
                P[:],_=stima_spettri(U[j,i,:],1/(Lx/Nx),Nwindows)
                pow_uu+=P
                Pc[:],_=stima_crossspettri(U[j,i,:],V[j,i,:],1/(Lx/Nx),Nwindows)
                pow_uv+=Pc
                Pc[:],_=stima_crossspettri(U[j,i,:],W[j,i,:],1/(Lx/Nx),Nwindows)
                pow_uw+=Pc
                Pc[:],_=stima_crossspettri(U[j,i,:],T[j,i,:],1/(Lx/Nx),Nwindows)
                pow_ut+=Pc

                P[:],_=stima_spettri(V[j,i,:],1/(Lx/Nx),Nwindows)
                pow_vv+=P
                Pc[:],_=stima_crossspettri(V[j,i,:],W[j,i,:],1/(Lx/Nx),Nwindows)
                pow_vw+=Pc
                Pc[:],_=stima_crossspettri(V[j,i,:],T[j,i,:],1/(Lx/Nx),Nwindows)
                pow_vt+=Pc

                P[:],_=stima_spettri(W[j,i,:],1/(Lx/Nx),Nwindows)
                pow_ww+=P
                Pc[:],_=stima_crossspettri(W[j,i,:],T[j,i,:],1/(Lx/Nx),Nwindows)
                pow_wt+=Pc

                P[:],_=stima_spettri(T[j,i,:],1/(Lx/Nx),Nwindows)
                pow_tt+=P

        pow_uu/=Ny*Nz
        pow_uv/=Ny*Nz
        pow_uw/=Ny*Nz
        pow_ut/=Ny*Nz

        pow_vv/=Ny*Nz
        pow_vw/=Ny*Nz
        pow_vt/=Ny*Nz

        pow_ww/=Ny*Nz
        pow_wt/=Ny*Nz

        pow_tt/=Ny*Nz

        return fr, pow_uu, pow_uv, pow_uw, pow_ut, pow_vv, pow_vw, pow_vt, pow_ww, pow_wt, pow_tt

    ###################################################################

    Nx=2**9
    Ny=2**6
    Nz=2**7 # This can be large without slowing down the code
    Lx=5120
    Ly=1280
    Lz=512
    
    # # isotropic test
    # alpha=0
    # Ri=0.0043
    # N=alpha*np.sqrt(Ri)
    # Gamma=3.78
    # Q_ratio=1
    # Us=0.49
    # T0=273
    # L=100

    # # Stratified test
    alpha=0.05
    Ri=0.14
    N=alpha*np.sqrt(Ri)
    Gamma=3.78
    Q_ratio=1
    Us=1
    T0=273
    L=0.4*100
    #####################################

    k1,k2,k3=get_wavenumbers(Nx,Ny,Nz,Lx,Ly,Lz)
    tt=time.time() 
    PHI=Compute_PHI(alpha,N,Us,L,k1,k2,k3) 
    print('execution time creation spectral tensor: ', time.time()-tt)

    tt=time.time()   
    U,V,W,T=Synthetic_turbulence(alpha,N,Us,T0,L,Nx,Ny,Nz,Lx,Ly,Lz,aliasing_correction=0,periodicity_correction=False)
    print('execution time creation time series: ', time.time()-tt)
    print('Variances: U,V,W,T ', np.var(U),np.var(V),np.var(W),np.var(T))

    Pxx,Pxy,Pxz,Pxt,Pyy,Pyz,Pyt,Pzz,Pzt,Ptt=get_spectrum_from_PHI(k1,PHI)
    fr, pow_uu, pow_uv, pow_uw, pow_ut, pow_vv, pow_vw, pow_vt, pow_ww, pow_wt, pow_tt = get_spectra_from_time_series(U,V,W,T,Nwindows=4)

    # savemat('example.mat',{'U':U,'V':V,'W':W,'T':T})
       
    ###################################################################

    plt.subplot(4,4,1)
    plt.semilogx(2*np.pi*fr,pow_uu*fr,'b')
    plt.semilogx(k1,Pxx*k1*Us**2,'r--')
    plt.grid()

    plt.subplot(4,4,2)
    plt.semilogx(2*np.pi*fr,np.real(pow_uv)*fr,'b')
    plt.semilogx(2*np.pi*fr,np.imag(pow_uv)*fr,'k')
    plt.semilogx(k1,np.real(Pxy)*k1*Us**2,'r--')
    plt.semilogx(k1,np.imag(Pxy)*k1*Us**2,'g--')
    plt.grid()

    plt.subplot(4,4,3)
    plt.semilogx(2*np.pi*fr,np.real(pow_uw)*fr,'b')
    plt.semilogx(2*np.pi*fr,np.imag(pow_uw)*fr,'k')
    plt.semilogx(k1,np.real(Pxz)*k1*Us**2,'r--')
    plt.semilogx(k1,np.imag(Pxz)*k1*Us**2,'g--')
    plt.grid()

    plt.subplot(4,4,4)
    plt.semilogx(2*np.pi*fr,np.real(pow_ut)*fr,'b')
    plt.semilogx(2*np.pi*fr,np.imag(pow_ut)*fr,'k')
    plt.semilogx(k1,np.real(Pxt)*k1*(T0*Us**2/(9.81*L))*Us,'r--')
    plt.semilogx(k1,np.imag(Pxt)*k1*(T0*Us**2/(9.81*L))*Us,'g--')
    plt.grid()

    plt.subplot(4,4,6)
    plt.semilogx(2*np.pi*fr,pow_vv*fr,'b')
    plt.semilogx(k1,Pyy*k1*Us**2,'r--')
    plt.grid()

    plt.subplot(4,4,7)
    plt.semilogx(2*np.pi*fr,np.real(pow_vw)*fr,'b')
    plt.semilogx(2*np.pi*fr,np.imag(pow_vw)*fr,'k')
    plt.semilogx(k1,np.real(Pyz)*k1*Us**2,'r--')
    plt.semilogx(k1,np.imag(Pyz)*k1*Us**2,'g--')
    plt.grid()

    plt.subplot(4,4,8)
    plt.semilogx(2*np.pi*fr,np.real(pow_vt)*fr,'b')
    plt.semilogx(2*np.pi*fr,np.imag(pow_vt)*fr,'k')
    plt.semilogx(k1,np.real(Pyt)*k1*(T0*Us**2/(9.81*L))*Us,'r--')
    plt.semilogx(k1,np.imag(Pyt)*k1*(T0*Us**2/(9.81*L))*Us,'g--')
    plt.grid()

    plt.subplot(4,4,11)
    plt.semilogx(2*np.pi*fr,pow_ww*fr,'b')
    plt.semilogx(k1,Pzz*k1*Us**2,'r--')
    plt.grid()

    plt.subplot(4,4,12)
    plt.semilogx(2*np.pi*fr,np.real(pow_wt)*fr,'b')
    plt.semilogx(2*np.pi*fr,np.imag(pow_wt)*fr,'k')
    plt.semilogx(k1,np.real(Pzt)*k1*(T0*Us**2/(9.81*L))*Us,'r--')
    plt.semilogx(k1,np.imag(Pzt)*k1*(T0*Us**2/(9.81*L))*Us,'g--')
    plt.grid()

    plt.subplot(4,4,16)
    plt.semilogx(2*np.pi*fr,pow_tt*fr,'b')
    plt.semilogx(k1,Ptt*k1*(T0*Us**2/(9.81*L))**2,'r--')
    plt.grid()

    plt.figure()
    plt.loglog(2*np.pi*fr,pow_uu*fr,'b')
    plt.loglog(k1[1:],Pxx[1:]*k1[1:]*Us**2,'r--')
    plt.loglog(k1[1:],(k1[1:]/L)**(-2/3)/80,'g--')
    plt.grid()
    
    plt.show()