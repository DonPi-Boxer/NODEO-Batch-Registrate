import argparse 
        
def create_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str,
                        dest="savedir", default="result",
                        help="Directory to save registration results to")
    parser.add_argument("--suffix_imgs", type=str,
                        dest="suffix_imgs", default="norm",
                        help="Suffix of filenames of the data images (without filetype name)")
    parser.add_argument("--suffix_segs", type=str,
                        dest="suffix_segs", default="seg",
                        help="Suffix of filenames of the data segmentation (without filetype name)")
    parser.add_argument("--statistics", type=int,
                        dest="statistics", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Save registration statistics")
    parser.add_argument("--visuals", type=int,
                        dest="visuals", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Save additional visuals")
    parser.add_argument("--label", type=str,
                        dest="label", default=[2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                        help="Segmentation labels to use during registration evaluation")
    parser.add_argument("--ds", type=int,
                        dest="ds", default=2,
                        help="specify output downsample times.")
    parser.add_argument("--bs", type=int,
                        dest="bs", default=16,
                        help="bottleneck size.")
    parser.add_argument("--smoothing", type=int,
                        dest="smoothing", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Perform smoothing")
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
    
    parser.add_argument("--device", type=str,
                       dest="device", default='cuda:0',
                       help="gpu: cuda:0; cpu: cpu")
   
    config = parser.parse_args()
    
    return config