import torch
import torch.nn.functional as F

'to do: rewrite jacobian, change x,y,z into x,y (for fig. 4)'

class NCC(torch.nn.Module):
    """
    NCC with cumulative sum implementation for acceleration. local (over window) normalized cross correlation.
    """

    def __init__(self, win=21, eps=1e-5):
        super(NCC, self).__init__()
        self.eps = eps
        self.win = win
        self.win_raw = win

    'changed *3 --> *2'
    def window_sum_cs2D(self, I, win_size):
        half_win = int(win_size / 2)
        pad = [half_win + 1, half_win] * 2

        I_padded = F.pad(I, pad=pad, mode='constant', value=0)  # [x+pad, y+pad]

        # Run the cumulative sum across all 3 dimensions
        I_cs_x = torch.cumsum(I_padded, dim=2)
        I_cs_xy = torch.cumsum(I_cs_x, dim=3)

        x, y = I.shape[2:]
        
        # Use subtraction to calculate the window sum
        I_win = I_cs_xy[:, :, win_size:, win_size:] \
                - I_cs_xy[:, :, win_size:, :y] \
                - I_cs_xy[:, :, :x, win_size:] \
                + I_cs_xy[:, :, :x, :y]
        return I_win

    def forward(self, I, J):
        # compute CC squares
        I = I.double()
        J = J.double()

        I2 = I * I
        J2 = J * J
        IJ = I * J

        'ref to 2D window sum func'
        # compute local sums via cumsum trick
        I_sum_cs = self.window_sum_cs2D(I, self.win)
        J_sum_cs = self.window_sum_cs2D(J, self.win)
        I2_sum_cs = self.window_sum_cs2D(I2, self.win)
        J2_sum_cs = self.window_sum_cs2D(J2, self.win)
        IJ_sum_cs = self.window_sum_cs2D(IJ, self.win)

        'replace **3 with **2'
        win_size_cs = (self.win * 1.) ** 2

        u_I_cs = I_sum_cs / win_size_cs
        u_J_cs = J_sum_cs / win_size_cs

        cross_cs = IJ_sum_cs - u_J_cs * I_sum_cs - u_I_cs * J_sum_cs + u_I_cs * u_J_cs * win_size_cs
        I_var_cs = I2_sum_cs - 2 * u_I_cs * I_sum_cs + u_I_cs * u_I_cs * win_size_cs
        J_var_cs = J2_sum_cs - 2 * u_J_cs * J_sum_cs + u_J_cs * u_J_cs * win_size_cs

        cc_cs = cross_cs * cross_cs / (I_var_cs * J_var_cs + self.eps)
        cc2 = cc_cs  # cross correlation squared

        # return negative cc.
        return 1. - torch.mean(cc2).float()

'changed to 2D, decrease amount of elements in J[] and see comments'
def JacboianDet(J):
    if J.size(-1) != 2:              #'3 --> 2'
        J = J.permute(0, 2, 3, 1)    #'was: permute(0,2,3,4,1)'
    J = J + 1
    J = J / 2.
    scale_factor = torch.tensor([J.size(1), J.size(2)]).to(J).view(1, 1, 1, 2) * 1.  #' was: view(1,1,1,1,3)'
    J = J * scale_factor

    'deleted dz'
    dy = J[:, 1:, :-1, :] - J[:, :-1, :-1, :]
    dx = J[:, :-1, 1:, :] - J[:, :-1, :-1, :]

    'new definition Jdet for 2D'
    Jdet = dx[:, :, :, 0] * dy[:, :, :, 1] - dx[:, :, :, 1] * dy[:, :, :, 0]

    return Jdet

def neg_Jdet_loss(J):
    Jdet = JacboianDet(J)
    neg_Jdet = -1.0 * (Jdet - 0.5)
    selected_neg_Jdet = F.relu(neg_Jdet)
    return torch.mean(selected_neg_Jdet ** 2)

def smoothloss_loss(df):
    return (((df[:, :, 1:, :] - df[:, :, :-1, :]) ** 2).mean() + \
     ((df[:, :, :, 1:] - df[:, :, :, :-1]) ** 2).mean())

def magnitude_loss(all_v):
    all_v_x_2 = all_v[:, :, 0, :, :] * all_v[:, :, 0, :, :]
    all_v_y_2 = all_v[:, :, 1, :, :] * all_v[:, :, 1, :, :]
    all_v_magnitude = torch.mean(all_v_x_2 + all_v_y_2)
    return all_v_magnitude


