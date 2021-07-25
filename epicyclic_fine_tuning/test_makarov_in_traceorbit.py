"""
Test epicyclic approximation:
compare results with galpy
"""
from __future__ import print_function, division

import numpy as np
import math
import sys
import os
import matplotlib.pyplot as plt
plt.ion()

from chronostar import traceorbit as torb
reload(torb)


# Pretty plots
from fig_settings import *


def many_timesteps(xyzuvw_start):
    #~ times = np.linspace(0, 25, 1000) # Myr
    times = np.linspace(0, 40, 1000) # Myr
    #~ times = np.linspace(0, 35, 1000) # Myr
    
    # Currently implemented in Chronostar
    #~ sA=0.89
    #~ sB=1.15
    #~ sR=1.21
    
    # Change this by little and it explodes!
    sA=0.89
    sB=1.15
    sR=1.21

    points_b=[]
    points_epi=[]
    for t in times:
        xyzuvw_b = torb.trace_cartesian_orbit(xyzuvw_start, times=t, single_age=True)
        #~ xyzuvw_epi = torb.trace_epicyclic_orbit(xyzuvw_start, times=t, sA=0.89, sB=1.15, sR=1.21)
        #~ xyzuvw_epi = torb.trace_epicyclic_orbit(xyzuvw_start, times=t) # sA, sB and sR are already in chronostar
        xyzuvw_epi = torb.trace_epicyclic_orbit(xyzuvw_start, times=t, sA=sA, sB=sB, sR=sR) # sA, sB and sR are already in chronostar
        points_b.append(xyzuvw_b)
        points_epi.append(xyzuvw_epi)
    points_b=np.array(points_b)
    points_epi=np.array(points_epi)
    
    # Plot how distances between Bovy and Epicyclic grow over time
    distance_pos = np.sqrt((points_b[:,0]-points_epi[:,0])**2 + (points_b[:,1]-points_epi[:,1])**2 + (points_b[:,2]-points_epi[:,2])**2)
    distance_vel = np.sqrt((points_b[:,3]-points_epi[:,3])**2 + (points_b[:,4]-points_epi[:,4])**2 + (points_b[:,5]-points_epi[:,5])**2)
    
    fig = plt.figure(figsize=(figsize[0], figsize[0]))
    ax=fig.add_subplot(111)
    ax.plot(times, distance_pos, c='k', label=r'$\Delta \mathrm{r}$')
    ax.set_xlabel('t [Myr]')
    ax.set_ylabel(r'$\Delta \mathrm{r \;[pc]}$')
    ax.set_ylim(0, 12)
    #~ ax.set_ylim(0, 3)
    
    
    ax2=ax.twinx()
    ax2.plot(times, distance_vel, c='r', label=r'$\Delta \mathrm{v}$')
    ax2.set_ylabel(r'$\Delta \mathrm{v \;[km \,s^{-1}]}$')
    ax2.set_ylim(0, 2)
    #~ ax2.set_ylim(0, 1)
    #~ ax2.set_xlim(0, 40)
    
    
    
    # LEGEND
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    labels = [labels[0], labels2[0]]
    handles = [handles[0], handles2[0]]
    legend=ax.legend(handles, labels, frameon=False) # , title=r'Distance between \texttt{galpy} and epicyclic orbits'

    plt.tight_layout()
    plt.savefig('compare_epicyclic_and_cartesian_orbits.pdf')

    # Positions over time
    #~ fig=plt.figure()
    #~ comb = [(0, 1), (0, 2), (2, 5), (3, 4)]
    #~ lnames = ['X', 'Y', 'Z', 'U', 'V', 'W', 'U', 'V']
    
    ##~ for i, (c, l) in enumerate(zip(comb, labels)):
    #~ for i, c in enumerate(comb):
        #~ ax=fig.add_subplot(2, 2, i+1)
        #~ ax.plot(points_b[:, c[0]], points_b[:, c[1]], c='r', label='Galpy')
        #~ ax.plot(points_epi[:, c[0]], points_epi[:, c[1]], c='b', label='Makarov')
        #~ ax.set_xlabel(lnames[c[0]])
        #~ ax.set_ylabel(lnames[c[1]])
    #~ plt.legend()
    #~ plt.tight_layout()
    ##~ plt.show()
  
def fine_tuning_A_B(xyzuvw_start):
    times = np.linspace(0, 35, 100) # Myr
    
    #~ sA = np.linspace(0.86, 0.88, 5)
    sA = np.linspace(0.87, 0.89, 2)
    #~ sB = np.linspace(1.15, 1.2, 5)
    sB = np.linspace(1.15, 1.17, 2)
    #~ sR = np.linspace(1.2, 1.4, 5)
    sR = np.linspace(1.19, 1.21, 2)


    # Galpy
    points_b=[]
    for t in times:
        xyzuvw_b = torb.trace_cartesian_orbit(xyzuvw_start, times=t, single_age=True)
        points_b.append(xyzuvw_b)
    points_b=np.array(points_b)

    # Epicyclic
    for A in sA:
        for B in sB:
            for Rho in sR:
                print(A, B, Rho)
                points_epi=[]
                for t in times:
                    xyzuvw_epi = torb.trace_epicyclic_orbit(xyzuvw_start, times=t, sA=A, sB=B, sR=Rho)
                    #~ print('FFF', xyzuvw_epi)
                    points_epi.append(xyzuvw_epi)
                points_epi=np.array(points_epi)

                fig=plt.figure()
                lnames = ['X', 'Y', 'Z', 'U', 'V', 'W', 'U', 'V']
                
                # Plot coordinates
                comb = [(0, 1), (0, 2), (2, 5), (3, 4)]
                for i, c in enumerate(comb):
                    ax=fig.add_subplot(2, 2, i+1)
                    ax.plot(points_b[:, c[0]], points_b[:, c[1]], c='r', label='Galpy')
                    ax.plot(points_epi[:, c[0]], points_epi[:, c[1]], c='b', label='Makarov')
                    ax.set_xlabel(lnames[c[0]])
                    ax.set_ylabel(lnames[c[1]])
                plt.suptitle('A=%f, B=%f, R=%f'%(A, B, Rho))
                plt.legend()
                plt.tight_layout()
   
def fine_tuning_A_B_parameter_plot(xyzuvw_start):
    times = np.linspace(0, 35, 100) # Myr
    
    #~ sA = np.linspace(0.86, 0.88, 5)
    #~ sB = np.linspace(1.15, 1.2, 5)
    #~ sR = np.linspace(1.2, 1.4, 5)
    
    #~ sA = np.linspace(0.87, 0.89, 10)
    #~ sB = np.linspace(1.15, 1.2, 5)
    #~ sR = np.linspace(1.19, 1.21, 20)
    
    #~ sA = np.linspace(0.85, 0.89, 10)
    #~ sB = np.linspace(1.15, 1.2, 10)
    #~ sR = np.linspace(1.19, 1.21, 20)
    
    #~ sA = np.linspace(0.5, 1.5, 20)
    sA = np.linspace(0.8, 1.3, 20)
    #~ sB = np.linspace(0.5, 1.5, 20)
    sR = np.linspace(1.2, 1.3, 20)
    #~ sR = np.linspace(0.7, 1.3, 20)
    
    sA = np.linspace(0.88, 1.91, 20)
    sB = np.linspace(1.14, 1.16, 20)
    sR = np.linspace(1.2, 1.3, 20)
    
    # So far, it seems that Rho 1.205 gives the best results

    # Galpy
    points_b=[]
    for t in times:
        xyzuvw_b = torb.trace_cartesian_orbit(xyzuvw_start, times=t, single_age=True)
        points_b.append(xyzuvw_b)
    points_b=np.array(points_b)

    

    # Epicyclic
    for Rho in sR:
        # Diff for XY, ZU and VW
        diff1=np.zeros((len(sA), len(sB)))
        diff2=np.zeros((len(sA), len(sB)))
        diff3=np.zeros((len(sA), len(sB)))
        for i, A in enumerate(sA):
            for j, B in enumerate(sB):
                points_epi=[]
                for t in times:
                    xyzuvw_epi = torb.trace_epicyclic_orbit(xyzuvw_start, times=t, sA=A, sB=B, sR=Rho)
                    points_epi.append(xyzuvw_epi)
                points_epi=np.array(points_epi)

                difference1 = points_epi[:, 1]-points_b[:, 1]
                difference2 = points_epi[:, 2]-points_b[:, 2]
                difference3 = points_epi[:, 5]-points_b[:, 5]
                difference1=np.abs(difference1)
                difference2=np.abs(difference2)
                difference3=np.abs(difference3)
                #~ print(difference1)
                #~ print(np.max(difference1))
                m1 = np.max(difference1)
                m2 = np.max(difference2)
                m3 = np.max(difference3)
                #~ print('')
                diff1[i, j] = m1
                diff2[i, j] = m2
                diff3[i, j] = m3
            
        diff1=np.array(diff1)
        diff2=np.array(diff2)
        diff3=np.array(diff3)
        # Kind of a measure...
        s = np.sqrt(diff1**2+diff2**2+diff3**2)
        s=np.log(s)

        fig=plt.figure(figsize=(8*2, 6*2))
        plt.suptitle('Rho %.3f'%Rho)
        
        ax=fig.add_subplot(223)
        diff1=np.log(diff1)
        cb=plt.imshow(diff1, aspect='auto', origin='lower', extent=[sA[0], sA[-1], sB[0], sB[-1]], vmax=1.5)
        plt.colorbar(cb)
        ax.set_xlabel('sA')
        ax.set_ylabel('sB')
        ax.set_title('Ygalpy-Yepi')
        
        ax=fig.add_subplot(221)
        cb=plt.imshow(diff2, aspect='auto', origin='lower', extent=[sA[0], sA[-1], sB[0], sB[-1]])
        plt.colorbar(cb)
        ax.set_xlabel('sA')
        #~ ax.set_ylabel('sB')
        ax.tick_params(labelleft=False)  
        ax.set_title('Zgalpy-Zepi')
        
        ax=fig.add_subplot(222)
        cb=plt.imshow(diff3, aspect='auto', origin='lower', extent=[sA[0], sA[-1], sB[0], sB[-1]])
        plt.colorbar(cb)
        ax.set_xlabel('sA')
        #~ ax.set_ylabel('sB')
        ax.tick_params(labelleft=False)  
        ax.set_title('Wgalpy-Wepi')
        
        ax=fig.add_subplot(224)
        cb=plt.imshow(s, aspect='auto', origin='lower', extent=[sA[0], sA[-1], sB[0], sB[-1]], vmax=3)
        plt.colorbar(cb)
        ax.set_xlabel('sA')
        #~ ax.set_ylabel('sB')
        ax.tick_params(labelleft=False)  
        ax.set_title('Sum')
        
        



if __name__ == '__main__':
    # A point from Chronostar's ScoCen run
    #~ xyzuvw_start = [80.97540887, -17.28686113, 46.73857814, 4.58384265, -7.03713712, 2.73081103]
    xyzuvw_start = np.array([133., -21., 48., -6., -17., -7]) # USco from Mamajek and Wright 2018
    #~ xyzuvw_start = np.array([13., -21., 48., -6., -17., -7]) # USco from Mamajek and Wright 2018
    #~ xyzuvw_start = [0,0,0,0,0,0]
    #~ xyzuvw_start = [-20,0,10,0,0,0]
    
    
    many_timesteps(xyzuvw_start)
    #~ fine_tuning_A_B(xyzuvw_start)
    #~ fine_tuning_A_B_parameter_plot(xyzuvw_start)
