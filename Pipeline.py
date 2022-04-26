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
from lenstronomy.Util import util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel

sys.path.insert(0, os.getcwd())
import utils

image_pre_path = './simulations/fits_files/i/'

name = int(sys.argv[1])

print(name-1)

modelized_systems = pd.read_csv('./test_dataset.csv')
#print(modelized_systems)
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

teste = utils.lens_light_test(cutout)
has_lens_light, f, x, y = teste.test()

if has_lens_light:

    #=========================== AutoLens ================================


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

    radius_value = mask_radius*float(pixel_scale)
    print(radius_value)
    if radius_value > 3.:
        radius_value = 3.
    print(radius_value)

    mask = al.Mask2D.circular(shape_native=imaging.shape_native, pixel_scales=pixel_scale, radius=radius_value)

    masked_object = imaging.apply_mask(mask=mask)

    # model
    source_galaxy_model = af.Model(al.Galaxy,
                                   redshift=zs)
    lens_bulge = af.Model(al.lmp.SphSersic)
    #lens_bulge = af.Model(al.lmp.EllSersic)
    lens_galaxy_model = af.Model(al.Galaxy,
                                 redshift=zl,
                                 bulge=lens_bulge)
    # combining our previous components
    lens_light_model = af.Collection(galaxies=af.Collection(lens=lens_galaxy_model, source=source_galaxy_model))
    print('Fit using Dynesty Static method...')

    #session=af.db.open_database("database.sqlite")
    search = af.DynestyStatic(path_prefix='./',
                              name = str(name),
                              unique_tag = 'LensLight_SPHSERSIC',
                              nlive = 50,
                              number_of_cores = 4) # be carefull here! verify your core numbers
                              #session=session) 

    analysis = al.AnalysisImaging(dataset=masked_object)
    step_0_result = search.fit(model=lens_light_model, analysis=analysis)

    image_output_autolens = (step_0_result.unmasked_model_image).reshape(100, 100)
    # rotate once
    rotate_image = utils.rotate_matrix(image_output_autolens)
    rotate_image = rotate_image.rotate()

    hdu = fits.PrimaryHDU(data=rotate_image)
    hdu.writeto('./fits_results/lens_light/'+str(name)+'_AutoLens[SPHSERSIC].fits')

    #=========================== ImFit ================================
    imfitConfigFile = "./config_imfit/config_galaxy.dat"
    model_desc = pyimfit.ModelDescription.load(imfitConfigFile)

    imfit_fitter = pyimfit.Imfit(model_desc)
    imfit_fitter.fit(cutout, gain=ccd_gain, read_noise=read_noise, original_sky=sky_rms, mask=mask)
    #imfit_fitter.fit(cutout, gain=ccd_gain, read_noise=read_noise, original_sky=sky_rms)
    
    if imfit_fitter.fitConverged is True:
        print('Fit converged: chi^2 = {0}, reduced chi^2 = {1}'.format(imfit_fitter.fitStatistic,
            imfit_fitter.reducedFitStatistic))
        bestfit_params = imfit_fitter.getRawParameters()
        print('Best-fit parameter values:', bestfit_params)
        f = open('./fits_results/lens_light/lens_light_report.csv', 'a') 
        f.write(str(name)+','+str(bestfit_params[0])+','+str(bestfit_params[1])+','+str(bestfit_params[2])+','+str(bestfit_params[3])+','+str(bestfit_params[4])+','+str(bestfit_params[5])+','+str(bestfit_params[6])+','+str(radius_value)+','+str(imfit_fitter.reducedFitStatistic)+'\n')

        f.close()

    hdu = fits.PrimaryHDU(data=imfit_fitter.getModelImage())
    hdu.writeto('./fits_results/lens_light/'+str(name)+'_ImFit[SPHSERSIC].fits')

    '''
    #=========================== Lenstronomy ================================
    # general configurations
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = util.make_grid_with_coordtransform(numPix=100, # horizontal (or vertical number of pixels) 
                                                                                            deltapix=pixel_scale, # pixel scale
                                                                                            center_ra=0, # lens ra position
                                                                                            center_dec=0, # lens dec position
                                                                                            subgrid_res=1, # resoluton factor of our images
                                                                                            inverse=False) # invert east to west?
    lens_light_model = ['SERSIC_ELLIPSE'] # mass distribution to our lens model. 

    # some fit parameters
    #lenstronomy_mask = np.invert(mask)*1
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

    # setting our image class
    # data input parameters
    kwargs_data = {'background_rms': sky_rms,  # rms of background in ADUs
                   'exposure_time': expo_time,  # exposure time
                   'ra_at_xy_0': ra_at_xy_0,  # RA at (0,0) pixel
                   'dec_at_xy_0': dec_at_xy_0,  # DEC at (0,0) pixel 
                   'transform_pix2angle': Mpix2coord,  # matrix to translate shift in pixel in shift in relative RA/DEC (2x2 matrix). Make sure it's units are arcseconds or the angular units you want to model.
                   'image_data': np.zeros((100, 100))}  # 2d data vector, here initialized with zeros as place holders that get's overwritten once a simulated image with noise is created.

    data_class = ImageData(**kwargs_data) 
    data_class.update_data(cutout)
    kwargs_data['image_data'] = cutout

    # setting our priors
    ######## Lens ########
    fixed_lens_light = []
    kwargs_lens_light_init = []
    kwargs_lens_light_sigma = []
    kwargs_lower_lens_light = []
    kwargs_upper_lens_light = []

    # initial guess, sigma, upper and lower parameters
    fixed_lens_light.append({})
    kwargs_lens_light_init.append({'R_sersic': 4., 'n_sersic': 2., 'e1': 0., 'e2': 0., 'center_x': 0, 'center_y': 0})
    kwargs_lens_light_sigma.append({'n_sersic': 2.5, 'R_sersic': 1., 'e1': 0.5, 'e2': 0.5, 'center_x': 1., 'center_y': 1.})
    kwargs_lower_lens_light.append({'e1': -1., 'e2': -1., 'R_sersic': 1., 'n_sersic': 0.1, 'center_x': -10, 'center_y': -10})
    kwargs_upper_lens_light.append({'e1': 1., 'e2': 1., 'R_sersic': 8., 'n_sersic': 8., 'center_x': 10, 'center_y': 10})

    # creating an object to have all this attributes
    lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, fixed_lens_light, kwargs_lower_lens_light, kwargs_upper_lens_light]
    kwargs_params = {'lens_light_model': lens_light_params}

    # Likelihood kwargs
    kwargs_likelihood = {'source_marg': False}
    kwargs_model = {'lens_light_model_list': lens_light_model} # Sersic, SIE, etc
    # here, we have 1 single band to fit
    multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]] # in this example, just a single band fit
    # if you have multiple  bands to be modeled simultaneously, you can append them to the mutli_band_list
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}  # 'multi-linear': every imaging band has independent solutions of the surface brightness, 'joint-linear': there is one joint solution of the linear coefficients demanded across the bands.
    # we dont have a constraint
    kwargs_constraints = {}

    # running an mcmc algorithm
    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)

    fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 200, 'n_iterations': 200}]]

    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()

    lens_light_model_class = LightModel(light_model_list=lens_light_model)

    imageModel = ImageModel(data_class, psf_class, lens_light_model_class=lens_light_model_class, kwargs_numerics=kwargs_numerics)
    image = imageModel.image(kwargs_lens_light=kwargs_result['kwargs_lens_light'])

    hdu = fits.PrimaryHDU(data=image)
    hdu.writeto('./fits_results/lens_light/'+str(name)+'_Lenstronomy[ELLSERSIC].fits')
    '''