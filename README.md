# Introduction
This repository is a project done for the course Deep-Learning on the Technical University of Delft. This repository is complementery to our reproducibility blogpost about the paper 
[CVPR 2022] NODEO: A Neural Ordinary Differential Equation Based Optimization Framework for Deformable Image Registration'.

Our reproducibility blogpost can be found here (TODO: put link to blogpost.....).
The original repostiroy of the original paper can be found here (TODO: link to NODEO-DIR repository).

This repository is a fork of the original respository, and focuses itself on two things.
First of all, this repository allows the user to perform the image registration over 2D images as well as over 3D images (while the original repository only supports 3D image registration).
Secondly, this paper allows for the execution of multiple registrations in batch. That is, if one wants to perform registration over multiple sets of images, one can simply run the script once, get a coffee and when he comes back all the desired registrations will be performed. This was also not possible in the original repository, which only allowed for the registration of two images within one run. 

# Usage
## Initialization 
* First clone this repo
```
git clone https://github.com/DonPi-Boxer/NODEO-DL-PROJECT.git
```
* Move into the directory NODEO-DL-PROJECT
```
cd NODEO-DL-PROJECT/
```

* Install the requirements. 
```
pip install -r requirements.txt
]
```
## Running the script
Usage for 2D and 3D batch-registration are almost identical. They should be performed within the 2D or 3D directory respectively.
* Move into the desired dimensionality directory
```
cd 3D/
```
OR

```
cd 2D/
```
Put the moving images and segmentations in the data/fixed directory.
Put the fixed images and segementations in the data/moving directory.
Default suffix for the images is 'norm', default suffix for the segmentation data is 'seg'.
Note that these suffices *do not* include the filetypes. 
So, by default, the filesnames of images end with 'norm.nii.gz' and the the filenames of the segmentations end with 'seg.nii.gz'.
Furthermore note that this script only supports image registration of 'nii.gz' files.

* Run the 'batch-registrate.py' file. 
  This can be done either from an IDE of choise, or from the CMD.

```
python3 batch-registrate.py 
```
Please note that, due to the addition of ArgParse arguments using the Boolean operator, the 'batch-registrate.py' file requires python3.9+ to be able to run succesfully.

##Displayed results
By default, registration results will be saved in the result/ directory. 
The result directory will be filled with directory of the form 'MOVING_IMAGE_NAME'/'FIXED_IMAGE_NAME'.
So, if e.g. one moving image is called moving and one fixed image is called fixed, the result of their registration will be saved in the directory
```
result/moving/fixed
```
By default, both the 2D and 3D batch registrations save a text file called 'statistics.txt' within these directories, displaying some evaluation statistics of this registration. Furthermore, once the entire batch registration is completed, a text file called 'batch-statistics' will be saved in the result directory, displaying the average evaluation of the registrations performed in batch.
By default, the 3D registration saves the deformation field in the result directory of the performed registration.

##Parameters 
Running the file from a command-line allows for the adaptation of following parameters. 

* These parameters can be shown in the CMD using 
 ```
 python3 batch-registrate.py --help
```
 *The following arguments can be given 

  ```                      [-h] [--savedir SAVEDIR]
                           [--suffix_imgs SUFFIX_IMGS]
                           [--suffix_segs SUFFIX_SEGS]
                           [--statistics | --no-statistics]
                           [--visuals | --no-visuals] [--label LABEL]
                           [--ds DS] [--bs BS] [--smoothing | --no-smoothing]
                           [--smoothing_kernel SMOOTHING_KERNEL]
                           [--smoothing_win SMOOTHING_WIN]
                           [--smoothing_pass SMOOTHING_PASS]
                           [--time_steps TIME_STEPS] [--optimizer OPTIMIZER]
                           [--STEP_SIZE STEP_SIZE] [--epoches EPOCHES]
                           [--NCC_win NCC_WIN] [--lr LR] [--lambda_J LAMBDA_J]
                           [--lambda_df LAMBDA_DF] [--lambda_v LAMBDA_V]
                           [--loss_sim LOSS_SIM] [--debug DEBUG]
                           [--device DEVICE]
```

The meaning of these arguments and their default values are as follows:
```
  -h, --help            show help message
  --savedir SAVEDIR     Directory to save registration results to
                        (Default: result)
  --suffix_imgs SUFFIX_IMGS
                        Suffix of filenames of the data images (without
                        filetype name)
                        (Default: 'norm')
  --suffix_segs SUFFIX_SEGS
                        Suffix of filenames of the data segmentation (without
                        filetype name)  
                        (Default: 'seg')
  --statistics, --no-statistics
                        Save registration statistics 
                        (Default: True)
  --visuals, --no-visuals
                        Save additional visuals 
                        (Default: True)
  --label LABEL         Segmentation labels to use during registration
                        evaluation
                        (Default: [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18])
  --ds DS               specify output downsample times.
                        (Default: 2)
  --bs BS               bottleneck size.
                        (Default: 16)
  --smoothing, --no-smoothing
                        Perform smoothing 
                        (Default: True)
  --smoothing_kernel SMOOTHING_KERNEL
                        AK: Averaging kernel; GK: Gaussian Kernel   Note: GK now only works in 3D
                        (Default: AK)
  --smoothing_win SMOOTHING_WIN
                        Smoothing Kernel size
  --smoothing_pass SMOOTHING_PASS
                        Number of Smoothing pass
                        (Default: 15)
  --time_steps TIME_STEPS
                        Number of time steps between the two images, >=2.
                        (Default: 1)
  --optimizer OPTIMIZER
                        Specify the optimizer to use. Euler or RK.
                        (Default: Euler)
  --STEP_SIZE STEP_SIZE
                        step size for numerical integration.
                        (Default: 0.001)
  --epoches EPOCHES     No. of epochs to train.
                        (Default: 300)
  --NCC_win NCC_WIN     NCC window size
                        (Default: 21)
  --lr LR               Learning rate.
                        (Default: 0.005)
  --lambda_J LAMBDA_J   Loss weight for neg J
                        (Default: 2.5)
  --lambda_df LAMBDA_DF
                        Loss weight for dphi/dx
                        (Default: 0.05)
  --lambda_v LAMBDA_V   Loss weight for neg J
                        (Default: 0.00005)
  --loss_sim LOSS_SIM   Similarity measurement to use   
                        (Default: NCC)
  --debug DEBUG         Open debug mode
  --device device       Device to run the registration on. GPU or CPU
                        (Default: GPU)
  ```

# Citation
@inproceedings{wu2022nodeo,
  title={Nodeo: A neural ordinary differential equation based optimization framework for deformable image registration},
  author={Wu, Yifan and Jiahao, Tom Z and Wang, Jiancong and Yushkevich, Paul A and Hsieh, M Ani and Gee, James C},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20804--20813},
  year={2022}
}

```js
const optionDefinitions = [
  { name: 'verbose', alias: 'v', type: Boolean },
  { name: 'src', type: String, multiple: true, defaultOption: true },
  { name: 'timeout', alias: 't', type: Number }
]
```
