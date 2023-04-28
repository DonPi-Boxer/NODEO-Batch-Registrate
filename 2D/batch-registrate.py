import os
import Registration
import config
import Save

#get and store all file paths for the MRI images and the segmentations in arrays
def main(config):
    #Get the moving and fixed file data
    moving_data_dir = 'data/moving'
    fixed_data_dir = 'data/fixed'
    moving_set_name_arr = []
    moving_file_paths_mri = []
    moving_file_paths_seg = []
    fixed_set_name_arr =[]
    fixed_files_paths_mri = []
    fixed_file_paths_seg = [] 
    for dirpath, dirnames, filenames in os.walk(moving_data_dir):
        for mriname in [f for f in filenames if f.endswith(config.suffix_imgs + ".nii.gz")]:
            moving_set_name_arr.append(os.path.basename(os.path.normpath(dirpath)))
            moving_file_paths_mri.append(os.path.join(dirpath,mriname))
        for segname in [f for f in filenames if f.endswith(config.suffix_segs + ".nii.gz")]:    
            moving_file_paths_seg.append(os.path.join(dirpath,segname)) 
    for dirpath, dirnames, filenames in os.walk(fixed_data_dir):
        for filename in [f for f in filenames if f.endswith(config.suffix_imgs + ".nii.gz")]:
            fixed_set_name_arr.append(os.path.basename(os.path.normpath(dirpath))) 
            fixed_files_paths_mri.append(os.path.join(dirpath,filename))
        for filename in [f for f in filenames if f.endswith(config.suffix_segs + ".nii.gz")]:
            fixed_file_paths_seg.append(os.path.join(dirpath,filename))
    #Placeholder variables for statistics if we want to output these
    if config.statistics:
        numruns = 0
        runtime = []
        mean_avg_dice = []
        mean_neg_j = []
        ratio_neg_j = []
    #Perform registration over the data paths
    for moving_set_name,moving_mri,moving_seg in zip(moving_set_name_arr,moving_file_paths_mri,moving_file_paths_seg):
        for fixed_set_name,fixed_mri,fixed_seg in zip(fixed_set_name_arr,fixed_files_paths_mri,fixed_file_paths_seg):
            if moving_mri != fixed_mri:
                numruns = numruns +1
                savedir_run = config.savedir + "/" + moving_set_name + '/' + fixed_set_name
                if not os.path.isdir(savedir_run):
                    os.makedirs(savedir_run)
                if config.visuals:
                    savedirvis = savedir_run + '/' + 'visuals'
                    if not os.path.isdir(savedirvis):
                        os.makedirs(savedirvis)
                avg_dice, runtime_run, mean_neg_j_run, ratio_neg_j_run = Registration.main(config = config, moving_mri = moving_mri, fixed_mri = fixed_mri,savedir=savedir_run, fixed_seg_in = fixed_seg, moving_seg_in=moving_seg)
               #If we want statistics, store them per run in the run savedir and store them to compute batch statistics later
                if config.statistics:
                    runtime.append(runtime_run)
                    mean_avg_dice.append(avg_dice)
                    mean_neg_j.append(mean_neg_j_run)
                    ratio_neg_j.append(ratio_neg_j_run)
                    Save.save_run_statistics(config, avg_dice, mean_neg_j_run, ratio_neg_j_run, runtime_run, savedir_run)
    #If we want statistics, save batch statistics to a textfile
    if config.statistics:
        Save.save_batch_statistics(config, mean_avg_dice,mean_neg_j, ratio_neg_j, runtime, numruns) 
        
if __name__ == '__main__':
    config = config.create_config()
    #If results folder does not yet exist, create one
    if not os.path.isdir(config.savedir):
            os.makedirs(config.savedir)
    main(config)