import numpy as np
import torch
import h5py
from scipy.fft import fft, ifft

def get_anthro_norm_params(hdf5_file):
    D_all_subjs = []
    X_all_subjs = []
    theta_all_subjs = []
    for subjects in hdf5_file:
        if len(hdf5_file[subjects]) > 4:
            this_subj = hdf5_file[subjects]
            D = this_subj.attrs['D']
            X = this_subj.attrs['X']
            theta = this_subj.attrs['theta']
            D_all_subjs.append(D)
            X_all_subjs.append(X)
            theta_all_subjs.append(theta)
    
    D_mu = np.mean(D_all_subjs, 0)
    X_mu = np.mean(X_all_subjs, 0)
    theta_mu = np.mean(theta_all_subjs, 0)

    D_std = np.std(D_all_subjs, 0)
    X_std = np.std(X_all_subjs, 0)
    theta_std = np.std(theta_all_subjs, 0)
    
    return D_mu, X_mu, theta_mu, D_std, X_std, theta_std


def get_lsd(pred, target, mean_op="all", frange=[0, 22050]):
    """
    Returns the LSD between db inputs 'pred' and the reference 'target'
    Input mean_op: "all"  - returns the mean of LSDs across all positions
                   "pos"  - returns LSD values for all positions
                   "freq" - returns mean of LSDs as a function of frequency
    """
    frange = torch.tensor(frange)
    fs = 44100
    diff = pred - target
    nfft = diff.size()[-3]*2
    fbin_min = torch.floor(frange[0]*nfft/fs).int()
    fbin_max = torch.minimum(torch.ceil(frange[1]*nfft/fs), torch.tensor(nfft/2-1)).int()
    diff = diff[:, fbin_min:fbin_max+1, :, :]
    if mean_op == "all":
        lsd = torch.mean(torch.sqrt(torch.mean(diff**2, -3)), (-2, -1))
    elif mean_op == "pos":
        lsd = torch.sqrt(torch.mean(diff**2, -3))
    elif mean_op == "freq":
        lsd = torch.sqrt(torch.mean(diff**2, (-2, -1)))
        
    
    return lsd

def get_log_fft(x, nfft=64, axis=-1, eps=1e-6):
    out = 2 * np.abs(fft(x, n=nfft, axis=axis)) / nfft
    out[out < 1e-6] = 1e-6
    out = np.log10(out)
    out = np.delete(out, np.arange(int(nfft/2), nfft), axis)
    
    return out

def linear_interpolate(in_hrtf, out_size, in_az_offset, in_el_offset):
    """
    Linearly interpolates "in_hrtf" along the spatial dimensions to size "out_size". 
    Input and output dimensions are [batch_size] x [channels] x [azimuth] x [elevation]
    Inputs:
        in_hrtf : Input HRTF
        out_size : Size of output
        in_az_offset : Offset of input az in the output 
        in_el_offset : Offset of input el in the output
    """
    
    
    
            