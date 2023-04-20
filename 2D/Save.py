import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from Utils import *

##Make this code such that we can run registration.py with visuals
def save_image(image, savedir, title):
    plt.figure()
    plt.title(title)
    plt.imshow(image)
    plt.savefig(savedir + '/' + 'visuals' + '/' + title + '.png')
    plt.close()

def save_grid(x,y,savedir, title, ax=None,  **kwargs):
    plt.figure(dpi=600)
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    kwargs.setdefault('linewidth', 0.1)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()
    # Make lines smaller
    plt.savefig(savedir + '/' + 'visuals' + '/' + title + '.png')
    
def save_result(warped_moving,savedir):
    save_nii(warped_moving.detach().cpu().numpy(), '%s/warped.nii.gz' % (savedir))

def save_df(savedir, df):
    save_nii(df.permute(2,3,0,1).detach().cpu().numpy(), '%s/df.nii.gz' % (savedir + '/visuals')) #'was: permute(2,3,4,0,1)