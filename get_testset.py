from spectral_cube import SpectralCube
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from scipy.optimize import curve_fit
from scipy import *
import time
import pprocess
from astropy.convolution import convolve
import radio_beam
import sys
from astropy.convolution import Gaussian1DKernel
from astropy import units as u
import numpy

def getres(freq = 110201.354 * u.MHz, diameter=12.*u.m):
	wavelength = freq.to(u.m, equivalencies=u.spectral())
	resolution = (1.22*wavelength) / (diameter) *u.rad # 12 meter NRAO telescope
	return resolution.to(u.arcsec)

def prep(f='/Users/jkeown/Desktop/DR21_13CO.fits', region='CygX_N', line='13CO'):
	if line=='13CO':
		freq = 330588. * u.MHz
	else:
		freq = 329331. * u.MHz
	header = fits.getheader(f)
	beam = getres(freq=freq, diameter=15.*u.m)
	header['BMIN'] = beam.to(u.deg).value # beam size in degrees
	header['BMAJ'] = beam.to(u.deg).value # beam size in degrees
	if region=='W3(OH)' or region=='W3Main' or region=='M16':	
		del header['CD1_2']
		del header['CD2_1']
	data = fits.getdata(f)
	data = data[1200:2200, :, :]
	header['NAXIS3'] = data.shape[0]
	fits.writeto(region+'_' + line + '_test.fits', data=data, header=header, overwrite=True)

	# If desired, convolve map with larger beam 
	# or load previously created convolved cube
	cube = SpectralCube.read(region+'_' + line + '_test.fits')
	cube_km_1 = cube.with_spectral_unit(u.km / u.s, velocity_convention='radio')
	beam = radio_beam.Beam(major=32*u.arcsec, minor=32*u.arcsec, pa=0*u.deg)
	cube_km = cube_km_1.convolve_to(beam)
	cube_km.write(region+'_' + line +'_conv_test.fits', format='fits', overwrite=True)

	#cube_km = SpectralCube.read(region+'_conv_test.fits')

	# Then smooth spectrally
	res = cube_km.spectral_axis[1] - cube_km.spectral_axis[2]
	new_axis = np.arange(cube_km.spectral_axis[-1].value,cube_km.spectral_axis[0].value,res.value * 2)*u.km/u.s
	fwhm_factor = np.sqrt(8*np.log(2))
	current_resolution = res
	target_resolution = res * 2
	pixel_scale = res
	gaussian_width = ((target_resolution**2 - current_resolution**2)**0.5 /
                  pixel_scale / fwhm_factor)
	kernel = Gaussian1DKernel(gaussian_width)
	new_cube = cube_km.spectral_smooth(kernel)
	interp_cube = new_cube.spectral_interpolate(new_axis,
                                          suppress_smooth_warning=True)
	interp_cube.write(region+'_' + line +'_conv_test_smooth.fits', overwrite=True)

def clipper(f = 'CygX_N_13CO_conv_test_smooth.fits', region='CygX_N', line='13CO'):
	data = fits.getdata(f)
	header = fits.getheader(f)
	#data = data[:,15:-15, 15:-15] # CygX_N
	#data = data[:,30:-60, 110:-15] # NGC7538
	#data = data[50:,110:-100, 250:-270] # B18
	data = data[600:1100,30:150, 40:145] # W3Main
	header['NAXIS2'] = data.shape[1]
	header['NAXIS1'] = data.shape[2]
	fits.writeto(region+'_' + line + '_conv_test_smooth_clip.fits', data=data, header=header, overwrite=True)
#prep(f='DR21_13CO.fits', region='CygX_N', line='13CO')
#clipper()
#prep(f='NGC7538_C18O.fits', region='NGC7538', line='C18O')
#clipper(f = 'NGC7538_C18O_conv_test_smooth.fits', region='NGC7538', line='C18O')
#clipper(f = 'B18_HC5N_base1.fits', region='B18', line='HC5N')
clipper(f = 'W3Main_C18O_conv_test_smooth.fits', region='W3Main', line='C18O')
