import numpy as np
from scipy.optimize import minimize

class find_radius:
    def __init__(self, pre_set_sigma, image_array):
        axis_integrated_image = np.sum(image_array, axis=0)
        over_mean_int_image=axis_integrated_image[50-pre_set_sigma:50+pre_set_sigma+1]
        self.y=over_mean_int_image/np.max(over_mean_int_image)
        self.x=np.linspace(50-pre_set_sigma, 50+pre_set_sigma, 2*pre_set_sigma+1, dtype=int)
    
    def gauss_func(self, x_val, norm_val, x0_val, sigma_val):
        return norm_val*np.exp(-0.5*((x_val-x0_val)/sigma_val)**2)

    def chi_squared(self, par):
        norm, x0, sigma = par
        gauss_array = self.gauss_func(self.x, norm, x0, sigma)
        return np.sum((self.y-gauss_array)**2)
    
    def get_radius(self, init_guess=[1., 50, 2], method='Nelder-Mead'):
        result=minimize(self.chi_squared, init_guess, method=method)
        return result.x
