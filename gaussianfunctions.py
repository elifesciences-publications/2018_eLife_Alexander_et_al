import scipy.optimize as opt
import numpy as np
import math

class modelValues():
    amp = None
    center_x = None
    center_y = None
    sigma_x = None
    sigma_y = None
    offset = None
    signal = None
    norm_height_guess = None

def twoD_Gaussian(x_data_tuple, amplitude, xo, yo, sigma_x, sigma_y, offset):
    """ Evaluate 2D Gaussian function based on x, y values and provided parameters. """
    (x, y) = x_data_tuple
    xo = float(xo)
    yo = float(yo)
    g = offset + amplitude*np.exp( - (((x-xo)/sigma_x)**2 + ((y-yo)/sigma_y)**2)/2)
    
    return g.ravel()

def tryGaussian(twoD_Gaussian, local_tuple, cur_frame_local_data, initial_guess, bound_tup):
    """ Attempt to fit 2D Gaussian """
    param = modelValues()
    try:
        popt, pcov = opt.curve_fit(twoD_Gaussian, local_tuple, cur_frame_local_data.ravel(), p0 = initial_guess, bounds=bound_tup)
        param.amp = popt[0]
        param.center_x = popt[1]
        param.center_y = popt[2]
        param.sigma_x = popt[3]
        param.sigma_y = popt[4]
        param.offset = popt[5]
        param.signal = 2 * math.pi *  param.amp * param.sigma_x * param.sigma_y
        param.norm_height_guess = (param.amp + param.offset) / param.offset


    except RuntimeError:
        param.signal = 0
        param.amp = 0
        param.norm_height_guess = 0
        param.center_x = np.nan
        param.center_y = np.nan
        param.sigma_x = 4
        param.sigma_y = 4
        param.offset = initial_guess[5]

    except ValueError:
        param.signal = 0
        param.amp = 0
        param.norm_height_guess = 0
        param.center_x = np.nan
        param.center_y = np.nan
        param.sigma_x = 4
        param.sigma_y = 4
        param.offset = initial_guess[5]

    return param

def getRSquared(twoD_Gaussian, data, par, size):
    """ Determine R^2 value for data and fit 2D Gaussian """
    x_val = np.linspace(0, size - 1, size)
    y_val = np.linspace(0, size - 1, size)
    x_val,y_val = np.meshgrid(x_val, y_val);
    g = twoD_Gaussian((x_val, y_val), par.amp, par.center_x, par.center_y, par.sigma_x, par.sigma_y, par.offset)
    ss_tot = ((data - np.mean(data))**2).sum()
    ss_res = ((data - g.reshape((size, size)))**2).sum()
    Rsqr = 1 - (ss_res/ss_tot)
    
    return Rsqr
