import numpy as np
import pandas as pd
from scipy.optimize import minimize
from astropy.io import fits
import os

class lens_light_test:
    def __init__(self, image_array):
        self.image_array=np.array(image_array)
    def test(self):
        image_bellow_zero=-cutout
        min_value=np.min(image_bellow_zero)
        [x_index], [y_index] = np.where(image_bellow_zero==min_value)
        if 45 < x_index< 55 and 45 < y_index< 55:
            return True, x_index, y_index
        else:
            return False, x_index, y_index
        
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
    
class getlenslight:
    def __init__(self, pre_path, cutout_name):
        
        method_dict = np.array(['ImFit', 'AutoLens', 'Lenstronomy'])
        self.method_dict = method_dict
        
        model_dict = np.array(['SPHSERSIC', 'ELLSERSIC', 'SPHSERSIC_nomask'])
        self.model_dict = model_dict
        
        self.pre_path = pre_path
        self.cutout_name = cutout_name
        
    def read_cutout(self, model, method):
        if model in self.model_dict and method in self.method_dict:
            return fits.open(str(self.pre_path)+str(self.cutout_name)+'_'+str(method)+'['+model+'].fits')[0].data
        else:
            raise Exception('Your model or method (or both) is/are not in the dictionaries. Please verify the class documentation.')
    def get_original_cutout(self):
        return fits.open('./simulations/fits_files/i/'+self.cutout_name+'.fits')[0].data

class getpickle:
    def __init__(self, pre_path, cutout_name):
        for root, dirs, files in os.walk(pre_path):
            if cutout_name in files:
                full_file_name = os.path.join(root, cutout_name)

        self.full_file_name = full_file_name
        
    def read_pickle(self):
        return pd.read_pickle(str(self.full_file_name))  

class rotate_matrix:
    def __init__(self, matrix):
        self.matrix = matrix
    def rotate(self):
        return [[self.matrix[jj][ii] for jj in range(len(self.matrix))] for ii in range(len(self.matrix[0])-1,-1,-1)]
    
