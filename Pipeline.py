import sys
import pandas as pd
import numpy as np
from astropy.io import fits
from lenstronomy.Data.psf import PSF
import autofit as af
import autolens as al
import autolens.plot as aplt
import os
import matplotlib.pyplot as plt
import pyimfit

sys.path.insert(0, os.getcwd())
import utils

image_pre_path = './simulations/fits_files/i/'

name = int(sys.argv[1])

print(name)

modelized_systems = pd.read_csv('./test_dataset.csv')
modelized_systems_select = modelized_systems[modelized_systems['OBJID-g'] == name]
modelized_systems_select.reset_index(inplace=True)

pixel_scale = float(modelized_systems_select['pixel_scale-g'][0])
seeing = float(modelized_systems_select['seeing-i'][0])
sky_rms = float(modelized_systems_select['sky_brightness-i'][0])
expo_time = float(modelized_systems_select['exposure_time-i'][0])
zl = float(modelized_systems_select['PLANE_1-REDSHIFT-g'][0])
zs = float(modelized_systems_select['PLANE_2-REDSHIFT-g'][0])
ccd_gain = float(modelized_systems_select['ccd_gain-i'][0])
read_noise = float(modelized_systems_select['read_noise-i'][0])
print(pixel_scale, seeing, sky_rms, expo_time, zl, zs, ccd_gain, read_noise)

cutout = fits.open(image_pre_path+str(int(name)-1)+'.fits')[0].data

kwargs_psf = {'psf_type': 'GAUSSIAN',
              'fwhm': seeing,
              'pixel_size': pixel_scale,
              'truncation': 4/seeing}
psf_class = PSF(**kwargs_psf)
psf = psf_class.kernel_point_source/np.max(psf_class.kernel_point_source)
#print(psf)
noise_map = np.sqrt((cutout*expo_time+float(sky_rms**2)))/expo_time

imaging = al.Imaging(al.Array2D.manual(np.array(cutout, dtype=float), pixel_scales=float(pixel_scale)), # cutout
                     al.Array2D.manual(np.array(noise_map, dtype=float), pixel_scales=float(pixel_scale)), # noise_map 
                     al.Kernel2D.manual(np.array(psf, dtype=float), pixel_scales=float(pixel_scale), shape_native=(100, 100))) # psf

finder = utils.find_radius(pre_set_sigma=3, image_array=cutout)
_, _, mask_radius= finder.get_radius() # gaussian normalization, center position (1-d) and radius (sigma)

radius_value = mask_radius*0.8*float(pixel_scale)
print(radius_value)
if radius_value > 3.:
    radius_value = 3.
print(radius_value)

mask = al.Mask2D.circular(shape_native=imaging.shape_native, pixel_scales=pixel_scale, radius=radius_value)
masked_object = imaging.apply_mask(mask=mask)

# model
source_galaxy_model = af.Model(al.Galaxy,
                               redshift=zs)
#lens_bulge = af.Model(al.lmp.SphSersic)
lens_bulge = af.Model(al.lmp.EllSersic)
lens_galaxy_model = af.Model(al.Galaxy,
                             redshift=zl,
                             bulge=lens_bulge)
# combining our previous components
lens_light_model = af.Collection(galaxies=af.Collection(lens=lens_galaxy_model, source=source_galaxy_model))
print('Fit using Dynesty Static method...')

#session=af.db.open_database("database.sqlite")
search = af.DynestyStatic(path_prefix='./',
                          name = str(name),
                          unique_tag = 'LensLight_ELLSERSIC',
                          nlive = 50,
                          number_of_cores = 4) # be carefull here! verify your core numbers
                          #session=session) 

analysis = al.AnalysisImaging(dataset=masked_object)
step_0_result = search.fit(model=lens_light_model, analysis=analysis)

hdu = fits.PrimaryHDU(data=(step_0_result.unmasked_model_image).reshape(100, 100))
hdu.writeto('./fits_results/lens_light/'+str(name)+'_AutoLens[ELLSERSIC].fits')

## ImFit
imfitConfigFile = "./config_imfit/config_galaxy.dat"
model_desc = pyimfit.ModelDescription.load(imfitConfigFile)

imfit_fitter = pyimfit.Imfit(model_desc)
imfit_fitter.fit(cutout, mask=mask.reshape(100, 100), gain=ccd_gain, read_noise=read_noise, original_sky=sky_rms)

if imfit_fitter.fitConverged is True:
    print("Fit converged: chi^2 = {0}, reduced chi^2 = {1}".format(imfit_fitter.fitStatistic,
        imfit_fitter.reducedFitStatistic))
    bestfit_params = imfit_fitter.getRawParameters()
    print("Best-fit parameter values:", bestfit_params)
    f = open("./fits_results/lens_light/lens_light_report_imfit.csv", "a") 
    f.write(str(name)+','+str(bestfit_params[0])+','+str(bestfit_params[1])+','+str(bestfit_params[2])+','+str(bestfit_params[3])+','+str(bestfit_params[4])+','+str(bestfit_params[5])+','+str(bestfit_params[6])+','+str(radius_value)+','+str(imfit_fitter.reducedFitStatistic)+'\n')
    
    f.close()

hdu = fits.PrimaryHDU(data=imfit_fitter.getModelImage())
hdu.writeto('./fits_results/lens_light/'+str(name)+'_ImFit[ELLSERSIC].fits')

'''
residual_image = al.Array2D.manual(np.array(cutout, dtype=float), pixel_scales=float(pixel_scale)) - step_0_result.unmasked_model_image

new_imaging = al.Imaging(residual_image, # cutout
                         al.Array2D.manual(np.array(noise_map, dtype=float), pixel_scales=float(pixel_scale)), # noise_map 
                         al.Kernel2D.manual(np.array(psf, dtype=float), pixel_scales=float(pixel_scale), shape_native=(100, 100))) # psf

new_mask = al.Mask2D.circular(shape_native=new_imaging.shape_native, pixel_scales=pixel_scale, radius=8.)
new_imaging = new_imaging.apply_mask(mask=new_mask)

bulge = af.Model(al.lmp.EllSersic)
source_galaxy_model = af.Model(al.Galaxy,
                               redshift=zs,
                               bulge=bulge)
# lens galaxy model
lens_galaxy_model = af.Model(al.Galaxy,
                             redshift=zl,
                             mass=al.mp.EllIsothermal)    

autolens_model = af.Collection(galaxies=af.Collection(lens=lens_galaxy_model, source=source_galaxy_model))

search = af.DynestyStatic(path_prefix = './',
                          name = str(name) + '_lens_light_step_1',
                          unique_tag = 'lenspop_deeplenstronomy_run0',
                          nlive = 30,
                          number_of_cores = 4) # be carefull here! verify your core numbers
                          #session=session) 

analysis = al.AnalysisImaging(dataset=new_imaging)
step_1_result = search.fit(model=autolens_model, analysis=analysis) # fbd = full bright distribution
'''