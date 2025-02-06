import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
import time
from scipy.ndimage import zoom
from contourpy import contour_generator

RADEG = 180/np.pi

def plot_fig(ifig):
        
    if ifig >= 2 and ifig <= 5:
        CHARSIZE = 1.0 #??
        N = 201
        Nt = 41
        Ne_i = 11
        run_id = 101
        it = 2
        Nspec = 101
        run_sort = 0
        
        if Nt == 11: 
            inc_lbl = ['i=87!Uo!N','i=82!Uo!N','i=77!Uo!N','i=71!Uo!N','i=66!Uo!N',
                       'i=60!Uo!N','i=54!Uo!N','i=47!Uo!N','i=39!Uo!N','i=30!Uo!N',
                       'i=17!Uo!N']
        elif Nt == 21:
            inc_lbl = ['i=89!Uo!N','i=86!Uo!N','i=83!Uo!N','i=80!Uo!N','i=78!Uo!N',
                       'i=75!Uo!N','i=72!Uo!N','i=69!Uo!N','i=66!Uo!N','i=63!Uo!N',
                       'i=60!Uo!N','i=57!Uo!N','i=53!Uo!N','i=50!Uo!N','i=46!Uo!N',
                       'i=42!Uo!N','i=38!Uo!N','i=33!Uo!N','i=28!Uo!N','i=22!Uo!N',
                       'i=12!Uo!N']
        elif Nt == 31:
            inc_lbl = ['i=89!Uo!N','i=87!Uo!N','i=85!Uo!N','i=84!Uo!N','i=82!Uo!N',
                       'i=80!Uo!N','i=78!Uo!N','i=76!Uo!N','i=74!Uo!N','i=72!Uo!N',
                       'i=70!Uo!N','i=68!Uo!N','i=66!Uo!N','i=64!Uo!N','i=62!Uo!N',
                       'i=60!Uo!N','i=58!Uo!N','i=56!Uo!N','i=53!Uo!N','i=51!Uo!N',
                       'i=49!Uo!N','i=46!Uo!N','i=43!Uo!N','i=41!Uo!N','i=38!Uo!N',
                       'i=35!Uo!N','i=31!Uo!N','i=27!Uo!N','i=23!Uo!N','i=18!Uo!N',
                       'i=10!Uo!N']
        elif Nt == 41:
            inc_lbl = ['i=89!Uo!N','i=88!Uo!N','i=87!Uo!N','i=85!Uo!N','i=84!Uo!N',
                       'i=82!Uo!N','i=81!Uo!N','i=79!Uo!N','i=78!Uo!N','i=77!Uo!N',
                       'i=75!Uo!N','i=74!Uo!N','i=72!Uo!N','i=71!Uo!N','i=69!Uo!N',
                       'i=68!Uo!N','i=66!Uo!N','i=65!Uo!N','i=63!Uo!N','i=62!Uo!N',
                       'i=60!Uo!N','i=58!Uo!N','i=57!Uo!N','i=55!Uo!N','i=53!Uo!N',
                       'i=52!Uo!N','i=50!Uo!N','i=48!Uo!N','i=46!Uo!N','i=44!Uo!N',
                       'i=42!Uo!N','i=40!Uo!N','i=38!Uo!N','i=35!Uo!N','i=33!Uo!N',
                       'i=30!Uo!N','i=27!Uo!N','i=24!Uo!N','i=20!Uo!N','i=16!Uo!N',
                       'i=9!Uo!N']
        
        if int(ifig) == 2: it = 0
        elif int(ifig) == 3: it = 5
        elif int(ifig) == 4: it = 10
        elif int(ifig) == 5: it = 20
        # if int(ifig) == 2: run_id = 100
        # elif int(ifig) == 3: run_id = 100
        # elif int(ifig) == 4: run_id = 102
        # elif int(ifig) == 5: run_id = 103
        run_id = 1250
        rdata = np.zeros((N,N,2))
        Ixy = np.zeros((N,N))
        wght = np.zeros((N,N))
        mov = np.zeros((Nt,N,N))
        movx = np.zeros((Nt,N,N))
        movy = np.zeros((Nt,N,N))
        smov = np.zeros((Nt,N,N,Ne_i))
        smovx = np.zeros((Nt,N,N,Ne_i))
        smovy = np.zeros((Nt,N,N,Ne_i))
        mov2 = np.zeros((N,N,Ne_i))
        movx2 = np.zeros((N,N,Ne_i))
        movy2 = np.zeros((N,N,Ne_i))
        spec = np.zeros((Nt,Nspec))
        Qspec = np.zeros((Nt,Nspec))
        Uspec = np.zeros((Nt,Nspec))
        spec_s = np.zeros((Nt,Nspec,6))
        Qspec_s = np.zeros((Nt,Nspec,6))
        Uspec_s = np.zeros((Nt,Nspec,6))
        spec2 = np.zeros((Nt,Nspec))
        Qspec2 = np.zeros((Nt,Nspec))
        Uspec2 = np.zeros((Nt,Nspec))
        deg_spec = np.zeros((Nt,Nspec))
        ang_spec = np.zeros((Nt,Nspec))

            
        rdatafile = f'data/scat_spec.{run_id}.dat'
        with open(rdatafile, 'r') as f:
            f_str = f.read()
            f_arr = re.split(r"[\s]+", f_str)[1:]
        
        start = 0
        spec[:] = np.array(f_arr[:spec.size]).reshape(spec.shape)
        start += spec.size
        Qspec[:] = np.array(f_arr[start:start+Qspec.size]).reshape(Qspec.shape)
        start += Qspec.size
        Uspec[:] = np.array(f_arr[start:start+Uspec.size]).reshape(Uspec.shape)
        start += Uspec.size

        for isort in range(6):
            spec2[:] = np.array(f_arr[start:start+spec2.size]).reshape(spec2.shape)
            start += spec2.size
            Qspec2[:] = np.array(f_arr[start:start+Qspec2.size]).reshape(Qspec2.shape)
            start += Qspec2.size
            Uspec2[:] = np.array(f_arr[start:start+Uspec2.size]).reshape(Uspec2.shape)
            start += Uspec2.size
            
            spec_s[:,:,isort] = spec2
            Qspec_s[:,:,isort] = Qspec2
            Uspec_s[:,:,isort] = Uspec2

        rdatafile = f'data/scat_imag.{run_id}.dat'
        with open(rdatafile, 'r') as f:
            f_str = f.read()
            f_arr = re.split(r"[\s]+", f_str)[1:]
        
        mov[:] = np.array(f_arr[:mov.size]).reshape(mov.shape)
        
        
        rdatafile = f'data/scat_ipol.{run_id}.dat'
        with open(rdatafile, 'r') as f:
            f_str = f.read()
            f_arr = re.split(r"[\s]+", f_str)[1:]
        movx[:] = np.array(f_arr[:movx.size]).reshape(movx.shape)
        movy[:] = np.array(f_arr[movx.size:movx.size+movy.size]).reshape(movy.shape)
        
        ## LOG ENERGY SPACING
        emin = 0.1
        # emin = 0.001 #AGN scale
        emax = 1000
        shorte = emin*10.**(np.arange(Nspec, dtype=float)/(Nspec-1.)*np.log10(emax/emin))
        bine = emin*10.**(np.arange(Ne_i+1, dtype=float)/(Ne_i)*np.log10(emax/emin))
        
        for i in range(Nt):
            mov[i,:,:]=np.transpose(mov[i,:,:])
            movx[i,:,:]=np.transpose(movx[i,:,:])
            movy[i,:,:]=np.transpose(movy[i,:,:])

        sortmov = np.sort(mov.flatten())
        movmax = np.max(mov)
        movmax=sortmov[1*N*N*Nt-100]
        # movmax=sortmov[1*N*N*Nt-1]
        outliers = np.argwhere(mov > movmax)
        if len(outliers):
            movx[outliers]=movx[outliers]/mov[outliers]*movmax
            movy[outliers]=movy(outliers)/mov[outliers]*movmax
            mov[outliers]=movmax

        y_min = 0
        je = run_sort 
        Ixy = mov[it,:,:]
        Xpol = movx[it,:,:]
        Ypol = movy[it,:,:]
        Ixy = Ixy+1e-10
        psi = np.arctan2(Ypol/Ixy,Xpol/Ixy)/2.
        tot_ang = np.arctan2(np.sum(Ypol),np.sum(Xpol))/2.*RADEG
        deg = np.sqrt((Ypol/Ixy)**2.+(Xpol/Ixy)**2.)
        tot_deg = np.sqrt(np.sum(Xpol)**2+np.sum(Ypol)**2)/np.sum(Ixy)
        Xpol = deg*np.cos(psi)
        Ypol = deg*np.sin(psi)
        Ixy = np.flip(Ixy,axis=0)
        Xpol = np.flip(Xpol,axis=0)
        Ypol = np.flip(Ypol,axis=0)
        I_data = 255*(np.log10(Ixy/movmax+1e-5)+5.01)/5.
        # data = (255*(Ixy/movmax)).tobytes()
        # print(np.max(data),tot_deg,tot_deg2,tot_ang,tot_ang2)
        N15 = (N-1)*1.5+1
        N2 = (int(600./N15))*N15
       
        Npx = 12600
        Nx = 10
        Ny = 5
        xx = np.arange(Nx, dtype='float')
        yy = np.arange(Ny, dtype='float')
        xscale = xx/(Nx-1.)*5.-5.
        xscale = 10.**xscale
        yscale = np.zeros(Nx)+1
        movscl = np.zeros((Nx,Ny))
        for i in range(Nx): movscl[i,:] = xscale[i]
        color_data = 255*(np.log10(movscl/np.max(movscl)+1e-5)+5.01)//5
        # color_data = zoom(color_data, [100, 1])
    
    
        # generate image  
        fig1, ax1 = plt.subplots()
        contour_plot = ax1.contourf(color_data, cmap='CMRmap', levels=80)
        plt.close(fig1)
        
        fig2, ax2 = plt.subplots()
        
        # polarization lines
        dd = 100.
        pstep=10
        maxIxy = np.max(Ixy)
        for i in range(0, N, pstep):
            for j in range(0, N, pstep):
                # x0 = i
                # y0 = j
                # dff = dd*(np.array([Ypol[j,i],Xpol[j,i],0]))
                dff = dd*(np.array([Xpol[j,i],Ypol[j,i],0]))
                if Ixy[j,i] > 1e-4*maxIxy:
                    # plt.plot([j-dff[0]/2.,j+dff[0]/2.], [i-dff[1]/2.,i+dff[1]/2.], color='white')
                    plt.plot([i-dff[0]/2.,i+dff[0]/2.], [j-dff[1]/2.,j+dff[1]/2.], color='white')

        
        # fig2, ax2 = plt.subplots()
        im = ax2.imshow(I_data, cmap='CMRmap')
        plt.xticks([])
        plt.yticks([])
            
        # plt.text(x=10, y=10, s='deg=5%', c='white', size=10)
        # plt.text(x=10, y=20, s=r'i=$75^o$', c='white', size=10)
        
        cbar1 = ax2.figure.colorbar(contour_plot, ax=ax2)
        cbar2 = ax2.figure.colorbar(im, ax=ax2)
        # ytickname=[r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$','1']
        # cbar.ax.set_yticklabels(ytickname)
        cbar1.ax.locator_params(nbins=5) 
        cbar1.set_label(r'$I/I_{max}$')
        
        plt.show()
        
    
def main():
    start = time.perf_counter()
    plot_fig(2)
    # for i in range(2, 6): plot_fig(i)
    end = time.perf_counter()
    print('time:', end-start)

    
main()
        