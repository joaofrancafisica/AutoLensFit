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

sys.path.insert(0, os.getcwd())
import utils

image_pre_path = './simulations/fits_files/i/'

name = int(sys.argv[1])

print(name)

modelized_systems = pd.read_csv('./modelized_systems.csv')
modelized_systems_select = modelized_systems[modelized_systems['OBJID-g'] == name]
modelized_systems_select.reset_index(inplace=True)

pixel_scale = float(modelized_systems_select['pixel_scale-g'][0])
seeing = float(modelized_systems_select['seeing-i'][0])
sky_rms = float(modelized_systems_select['sky_brightness-i'][0])
expo_time = float(modelized_systems_select['exposure_time-i'][0])
zl = float(modelized_systems_select['PLANE_1-REDSHIFT-g'][0])
zs = float(modelized_systems_select['PLANE_2-REDSHIFT-g'][0])
print(pixel_scale, seeing, sky_rms, expo_time, zl, zs)



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
print(mask_radius)

radius_value = mask_radius*0.8*float(pixel_scale)
if mask_radius*0.8*float(pixel_scale) > 8.:
    radius_value = 8.

mask = al.Mask2D.circular(shape_native=imaging.shape_native, pixel_scales=pixel_scale, radius=radius_value)
masked_object = imaging.apply_mask(mask=mask)

# model
source_galaxy_model = af.Model(al.Galaxy,
                               redshift=zs)
lens_bulge = af.Model(al.lmp.EllSersic)
lens_galaxy_model = af.Model(al.Galaxy,
                             redshift=zl,
                             bulge=lens_bulge)
# combining our previous components
lens_light_model = af.Collection(galaxies=af.Collection(lens=lens_galaxy_model, source=source_galaxy_model))
print('Fit using Dynesty Static method...')

session=af.db.open_database("database.sqlite")
search = af.DynestyStatic(path_prefix='./',
                          name = str(name) + '_lens_light_step_0',
                          unique_tag = 'lenspop_deeplenstronomy_run0',
                          nlive = 30,
                          number_of_cores = 4, # be carefull here! verify your core numbers
                          session=session) 

analysis = al.AnalysisImaging(dataset=masked_object)
step_0_result = search.fit(model=lens_light_model, analysis=analysis)

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
                          number_of_cores = 4, # be carefull here! verify your core numbers
                          session=session) 

analysis = al.AnalysisImaging(dataset=new_imaging)
step_1_result = search.fit(model=autolens_model, analysis=analysis) # fbd = full bright distribution