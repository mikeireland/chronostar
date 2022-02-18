# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as sp

g, bp , rp, age, bn, feh, m, ms = np.load('10MSTAR_POPULATION_GBRABFM.npy').T

def show_pop():
    fig, ax = plt.subplots()
    im = ax.scatter((bp-rp),g,c=age,s=1, alpha=0.9)
    fig.colorbar(im, ax=ax)
    ax.set_ylabel('G')
    ax.set_xlabel('Gaia B-R')
    ax.set_ylim(15.5,-10)
    ax.figure.set_size_inches(4, 7)
    fig.savefig('KerSynthfc.png',dpi=200)
    fig.savefig('KerSynthfc.pdf')
    fig.show1
    
zipnowork=np.vstack(((bp-rp),g)).T
zipnowork.shape
tree=sp.KDTree(zipnowork)

#%%
def treeGetAges(col,mag,radius, data=tree):

    inds=data.query_ball_point((col,mag),radius)
    ages=age[inds]
    colres=bp[inds]-rp[inds]
    gres=g[inds]

    return ages, colres, gres

lgage=np.arange(5,11.4, 0.1)

def make_hists(col, gmag, n=100, r=0.01, data=tree):
    ages,colres,gres=treeGetAges(col, gmag, r, data=data)
    if len(ages)>(n*2):
        r=0.75*r
        ages,colres,gres=treeGetAges(col, gmag, r, data=data)
    if len(ages)<n:
        r=1.5*r
        ages,colres,gres=treeGetAges(col, gmag, r, data=data)
    
    dists=np.sqrt((col-colres)**2+(gmag-gres)**2)
    ages=ages[np.argsort(dists)]
    ages=ages[:n]
    ags=np.round((ages-5)/0.1).astype(int)
    age_hists=np.zeros_like(lgage)
    for i in ags:
        age_hists[i]+=1
    return age_hists    
    
gaus  = np.exp(-(lgage-np.median(lgage))**2/ 0.1**2 /2)
gaus /= np.sum(gaus)
gausft = np.fft.rfft(np.fft.fftshift(gaus))

def g_kernal_den(col, gmag, n=100, r=0.01, data=tree, 
                 show_PDF=False, show_NearPop=True):
    age_h=make_hists(col, gmag, n=n, r=r, data=data);
    age_pdf=np.fft.irfft(np.fft.rfft(age_h*10**lgage)*gausft);
    grated=np.trapz(age_pdf,10**lgage);
    normed=age_pdf/grated;
    if show_PDF:
         fig, ax = plt.subplots()
         ax.plot(lgage,normed)
         ax.set_ylabel('Relative Probability')
         ax.set_xlabel('log(Age)')
         fig.savefig('g_kernal_den_OUT.pdf')
         fig.show
    return normed

def get_probage(A, pdf, # a pdf is made from g_kernal_den
                n=100, r=0.01, data=tree, 
                show_PDF=False, show_NearPop=False):
     if not(5<A<11.4):
        print('Err; log(age) out of 5 to 11.4 interval')
     
     for i in range(len(lgage)):
         if lgage[i+1] > A:
             grad=(pdf[i+1]-pdf[i])/(lgage[i+1]-lgage[i])
             prob=grad*(A-lgage[i])+pdf[i]
             break
        
     return prob