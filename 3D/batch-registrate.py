import os
import sys
import glob
import Registration
import argparse
import subprocess
import numpy as np

#get and store all file paths for the MRI images and the segmentations in arrays

def main(config):
    moving_data_dir = 'data/moving'
    fixed_data_dir = 'data/fixed'
    moving_set_name_arr = []
    moving_file_paths_mri = []
    moving_file_paths_seg = []
    fixed_set_name_arr =[]
    fixed_files_paths_mri = []
    fixed_file_paths_seg = [] 
    for dirpath, dirnames, filenames in os.walk(moving_data_dir):
        for mriname in [f for f in filenames if f.endswith(config.suffix_imgs +  ".nii.gz")]:
            moving_set_name_arr.append(os.path.basename(os.path.normpath(dirpath)))
            moving_file_paths_mri.append(os.path.join(dirpath,mriname))
        for segname in [f for f in filenames if f.endswith(config.suffix_segs + ".nii.gz")]:    
            moving_file_paths_seg.append(os.path.join(dirpath,segname))
    
    for dirpath, dirnames, filenames in os.walk(fixed_data_dir):
        for filename in [f for f in filenames if f.endswith(config.suffix_imgs +  ".nii.gz")]:
            fixed_set_name_arr.append(os.path.basename(os.path.normpath(dirpath))) 
            fixed_files_paths_mri.append(os.path.join(dirpath,filename))
        for filename in [f for f in filenames if f.endswith(config.suffix_segs + ".nii.gz")]:
            fixed_file_paths_seg.append(os.path.join(dirpath,filename))
    numruns = 0
    runtime = []
    mean_avg_dice = []
    mean_neg_j = []
    ratio_neg_j = []
    for moving_set_name,moving_mri,moving_seg in zip(moving_set_name_arr,moving_file_paths_mri,moving_file_paths_seg):
        for fixed_set_name,fixed_mri,fixed_seg in zip(fixed_set_name_arr,fixed_files_paths_mri,fixed_file_paths_seg):
            if moving_mri != fixed_mri:
                numruns = numruns +1
                savedir = './result/' + moving_set_name +'/' + fixed_set_name
                print(savedir)
                if not os.path.isdir(savedir):
                    os.makedirs(savedir)
                avg_dice, runtime_run, mean_neg_j_run, ratio_neg_j_run = Registration.main(config = config, moving_mri = moving_mri, fixed_mri = fixed_mri,savedir=savedir, fixed_seg_in = fixed_seg, moving_seg_in=moving_seg)
                runtime.append(runtime_run)
                mean_avg_dice.append(avg_dice)
                mean_neg_j.append(mean_neg_j_run)
                ratio_neg_j.append(ratio_neg_j_run)
                print("Mean average dice after " , numruns , " registrations is " , np.mean(mean_avg_dice))
                print("Mean total negjet after " , numruns , " registrations is " , np.mean(mean_neg_j))
                print("Ratio negjet after " , numruns , " registrations is " , np.mean(ratio_neg_j))
    #print to terminale, now do this double as im ensure if subseuqent saving works
    print("all done !")
    print("Mean avg dice is ", np.mean(mean_avg_dice), "with an std of ", np.std(mean_avg_dice))
    print("Mean total negjet is", np.mean(mean_neg_j), "with an std of ", np.std(mean_neg_j))
    print("Mean ratio negjes is ", np.mean(ratio_neg_j), " with an std of ", np.std(ratio_neg_j))
    print("total runtime was ", np.sum(runtime), " for in total ", numruns, " Registrations")
    print("So average runtime was ", np.mean(runtime))
    original_stdout = sys.stdout
    #print to terminal and save to a textfile
    with open('performance.txt', 'w') as f:
        sys.stdout = f
        print("all done !")
        print("Mean avg dice is ", np.mean(mean_avg_dice), "with an std of ", np.std(mean_avg_dice))
        print("Mean total negjet is", np.mean(mean_neg_j), "with an std of ", np.std(mean_neg_j))
        print("Mean ratio negjes is ", np.mean(ratio_neg_j), " with an std of ", np.std(ratio_neg_j))
        print("total runtime was ", np.sum(runtime), " for in total ", numruns, " Registrations")
        print("So average runtime was ", np.mean(runtime))
        sys.stdout = original_stdout
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix_imgs", type=str,
                        dest="suffix_imgs", default="norm",
                        help="Suffix of filenames of the data images (without filetype name)")
    parser.add_argument("--suffix_segs", type=str,
                        dest="suffix_segs", default="seg",
                        help="Suffix of filenames of the data segmentation (without filetype name)")
    parser.add_argument("--visuals", type=int,
                        dest="visuals", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Save additional visuals")
    parser.add_argument("--ds", type=int,
                        dest="ds", default=2,
                        help="specify output downsample times.")
    parser.add_argument("--bs", type=int,
                        dest="bs", default=16,
                        help="bottleneck size.")
    parser.add_argument("--smoothing_kernel", type=str,
                        dest="smoothing_kernel", default='AK',
                        help="AK: Averaging kernel; GK: Gaussian Kernel")
    parser.add_argument("--smoothing_win", type=int,
                        dest="smoothing_win", default=15,
                        help="Smoothing Kernel size")
    parser.add_argument("--smoothing_pass", type=int,
                        dest="smoothing_pass", default=1,
                        help="Number of Smoothing pass")
    # Training configuration
    parser.add_argument("--time_steps", type=int,
                        dest="time_steps", default=2,
                        help="number of time steps between the two images, >=2.")
    parser.add_argument("--optimizer", type=str,
                        dest="optimizer", default='Euler',
                        help="Euler or RK.")
    parser.add_argument("--STEP_SIZE", type=float,
                        dest="STEP_SIZE", default=0.001,
                        help="step size for numerical integration.")
    parser.add_argument("--epoches", type=int,
                        dest="epoches", default=300,
                        help="No. of epochs to train.")
    parser.add_argument("--NCC_win", type=int,
                        dest="NCC_win", default=21,
                        help="NCC window size")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=0.005,
                        help="Learning rate.")
    parser.add_argument("--lambda_J", type=int,
                        dest="lambda_J", default=2.5,
                        help="Loss weight for neg J")
    parser.add_argument("--lambda_df", type=int,
                        dest="lambda_df", default=0.05,
                        help="Loss weight for dphi/dx")
    parser.add_argument("--lambda_v", type=int,
                        dest="lambda_v", default=0.00005,
                        help="Loss weight for neg J")
    parser.add_argument("--loss_sim", type=str,
                        dest="loss_sim", default='NCC',
                        help="Similarity measurement")
    # Debug
    parser.add_argument("--debug", type=bool,
                        dest="debug", default=False,
                        help="debug mode")
    #Device run on GPU
    parser.add_argument("--device", type=str,
                        dest="device", default='cuda:0',
                        help="gpu: cuda:0; cpu: cpu")
    # Device run on CPU    
    parser.add_argument("--device", type=str,
                        dest="device", default='cpu')
    config = parser.parse_args()
    
main(config)