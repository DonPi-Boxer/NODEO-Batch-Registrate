import sys
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
    
#Save statistics of this run to a text file  
def save_run_statistics(config, av_dice, mean_neg_J, ratio_neg_J, runtime, savedir):
    original_stdout = sys.stdout
    with open(savedir + '/' + 'statistics.txt', 'w') as f:
        sys.stdout = f
        print('Total of neg Jet: ', mean_neg_J)
        print('Ratio of neg Jet: ', ratio_neg_J)
        print('Avg. dice on %d structures: ' % len(config.label), av_dice)
        print("Runtime was: ", runtime)
        sys.stdout = original_stdout

def save_batch_statistics(config, av_dice, mean_neg_J, ratio_neg_J, runtime, numruns):
    original_stdout = sys.stdout
    with open(config.savedir + '/batch-statistics.txt', 'w') as f:
        sys.stdout = f
        print("Mean avg dice is ", np.mean(av_dice), "with an std of ", np.std(av_dice))
        print("Mean total negjet is", np.mean(mean_neg_J), "with an std of ", np.std(mean_neg_J))
        print("Mean ratio negjet is ", np.mean(ratio_neg_J), " with an std of ", np.std(ratio_neg_J))
        print("total runtime was ", np.sum(runtime), " for in total ", numruns, " Registrations")
    sys.stdout = original_stdout