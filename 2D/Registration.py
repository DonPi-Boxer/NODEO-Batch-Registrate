import time
import Loss
from Network import BrainNet
from Loss import *
from NeuralODE import *
from Utils import *
import Save
'changes made for 2D'
'to make the results of fig.4, we added them to the main function'

'"main" defines what to do with config'
'input is config, this is the configuration specification defined at bottom of this script'

def main(config, moving_mri, fixed_mri, savedir, fixed_seg_in, moving_seg_in):
    device = torch.device(config.device)  ## config.device: now set to 'CPU'
    #CHANGED BY NJ
    fixed = load_nii(fixed_mri)        ## load the fixed image(s)
    moving = load_nii(moving_mri)      ## and moving image(s)
    assert fixed.shape == moving.shape    # two images to be registered must in the same size
    # Plot moving and fixed as a 2d image
    t = time.time()
    df, df_with_grid, warped_moving = registration(config, device, moving, fixed)
    runtime = time.time() - t
    print('Registration Running Time:', runtime)
    print('---Registration DONE---')
    av_dice, mean_neg_j, ratio_neg_j, neg_Jdet = evaluation(config,device, df, df_with_grid, fixed_seg_in, moving_seg_in)
    if config.visuals:
        Save.save_image(moving,savedir, 'Moving Image')
        Save.save_image(fixed, savedir, 'Fixed Image')
        Save.save_df(savedir, df)
        # Plot warped as a 2D image
        warped_moving_np = warped_moving.detach().cpu().numpy()
        Save.save_image(warped_moving_np, savedir,'Warped Moving Image')
        df_with_grid_np = df_with_grid.cpu().numpy()
        # Plot the 2D deformation field
        # Save deformation field with grid
        Save.save_grid(df_with_grid_np[0, :, :, 0], df_with_grid_np[0, :, :, 1], savedir,'df with grid')
        # Plot the jacobian determinants of the deformation field
        jdet = Loss.JacboianDet(df_with_grid)
        Save.save_image(jdet.cpu().numpy()[0, :, :],savedir, "Jacobian Determinant")
        # Plot negative jacobian determinant
        Save.save_image(neg_Jdet.cpu().numpy()[0, :, :],savedir, "Negative Jacobian Determinant")   
    print('---Evaluation DONE---')
    Save.save_result(warped_moving,savedir)
    print('---Results Saved---')
    return av_dice, runtime, mean_neg_j, ratio_neg_j


'"registration defines the process of going through the neural network'
def registration(config, device, moving, fixed):
    '''
    Registration moving to fixed.
    :param config: configurations.
    :param device: gpu or cpu.
    :param img1: moving image to be registered, geodesic shooting starting point.
    :param img2: fixed image, geodesic shooting target.
    :return ode_train: neuralODE class.
    :return all_phi: Displacement field for all time steps.
    '''
    im_shape = fixed.shape
    moving = torch.from_numpy(moving).to(device).float()
    fixed = torch.from_numpy(fixed).to(device).float()
    # make batch dimension
    moving = moving.unsqueeze(0).unsqueeze(0)
    fixed = fixed.unsqueeze(0).unsqueeze(0)

    'define the network by using the class "BrainNet" from the "network" script'
    'BrainNet with specific inputs img_sz etc. from "config"'
    Network = BrainNet(img_sz=im_shape,
                       smoothing=config.smoothing,
                       smoothing_kernel=config.smoothing_kernel,
                       smoothing_win=config.smoothing_win,
                       smoothing_pass=config.smoothing_pass,
                       ds=config.ds,
                       bs=config.bs
                       ).to(device)

    'use class "NeuralODE" from script "NeuralODE"'
    ode_train = NeuralODE(Network, config.optimizer, config.STEP_SIZE).to(device)

    'changed: 3-->2 and removed a 1 at the end'
    # training loop
    scale_factor = torch.tensor(im_shape).to(device).view(1, 2, 1, 1) * 1.
    ST = SpatialTransformer(im_shape).to(device)  # spatial transformer to warp image
    grid = generate_grid2D_tensor(im_shape).unsqueeze(0).to(device)  # [-1,1]

    # Define optimizer
    optimizer = torch.optim.Adam(ode_train.parameters(), lr=config.lr, amsgrad=True)
    loss_NCC = NCC(win=config.NCC_win)
    BEST_loss_sim_loss_J = 1000
    for i in range(config.epoches):
        all_phi = ode_train(grid, Tensor(np.arange(config.time_steps)), return_whole_sequence=True)
        all_v = all_phi[1:] - all_phi[:-1]
        all_phi = (all_phi + 1.) / 2. * scale_factor  # [-1, 1] -> voxel spacing
        phi = all_phi[-1]
        grid_voxel = (grid + 1.) / 2. * scale_factor  # [-1, 1] -> voxel spacing
        #df = phi 'probeersel'
        df = phi - grid_voxel  # with grid -> without grid
        warped_moving, df_with_grid = ST(moving, df, return_phi=True)
        # similarity loss
        loss_sim = loss_NCC(warped_moving, fixed)
        warped_moving = warped_moving.squeeze(0).squeeze(0)
        # V magnitude loss
        loss_v = config.lambda_v * magnitude_loss(all_v)
        # neg Jacobian loss
        loss_J = config.lambda_J * neg_Jdet_loss(df_with_grid)
        # phi dphi/dx loss
        loss_df = config.lambda_df * smoothloss_loss(df)
        ' Regularisation term consists of above three terms'
        ' To make table 4 simply remove loss_J from the "loss" expression: '
        #loss = loss_sim + loss_v + loss_df
        loss = loss_sim + loss_v + loss_J + loss_df
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 20 == 0:
            print("Iteration: {0} Loss_sim: {1:.3e} loss_J: {2:.3e}".format(i + 1, loss_sim.item(), loss_J.item()))
        # pick the one df with the most balance loss_sim and loss_J in the last 50 epoches
        if i > config.epoches - 50:
            loss_sim_loss_J = 1000 * loss_sim.item() * loss_J.item()
            if loss_sim_loss_J < BEST_loss_sim_loss_J:
                best_df = df.detach().clone()
                best_df_with_grid = df_with_grid.detach().clone()
                best_warped_moving = warped_moving.detach().clone()
    return best_df, best_df_with_grid, best_warped_moving

'utensil for "main"'
def evaluation(config,device, df, df_with_grid, fixed_seg_in, moving_seg_in):
    ### Calculate Neg Jac Ratio
    neg_Jet = -1.0 *  JacboianDet(df_with_grid)
    neg_Jet = F.relu(neg_Jet)
    mean_neg_J = torch.sum(neg_Jet).detach().cpu().numpy()
    num_neg = len(torch.where(neg_Jet > 0)[0])
    total = neg_Jet.size(-1) * neg_Jet.size(-2) * neg_Jet.size(-3)
    ratio_neg_J = num_neg / total
    print('Total of neg Jet: ', mean_neg_J)
    print('Ratio of neg Jet: ', ratio_neg_J)
    ### Calculate Dice
    label = config.label
    fixed_seg = load_nii(fixed_seg_in)
    moving_seg = load_nii(moving_seg_in)
    ST_seg = SpatialTransformer(fixed_seg.shape, mode='nearest').to(device)
    moving_seg = torch.from_numpy(moving_seg).to(device).float()
    # make batch dimension
    moving_seg = moving_seg[None, None, ...]
    warped_seg = ST_seg(moving_seg, df, return_phi=False)
    dice_move2fix = dice(warped_seg.unsqueeze(0).unsqueeze(0).detach().cpu().numpy(), fixed_seg, label)
    av_dice = np.mean(dice_move2fix[0])
    print('Avg. dice on %d structures: ' % len(label), av_dice)
    return av_dice, mean_neg_J, ratio_neg_J, neg_Jet
   


