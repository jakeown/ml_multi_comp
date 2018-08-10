import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits
import numpy
import numpy as np
from scipy.stats import multivariate_normal
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
import h5py

x, y, z = np.mgrid[-1.0:1.0:500j, -1.0:1.0:10j, -1.0:1.0:10j]

# Need an (N, 2) array of (x, y) pairs.
xy = np.column_stack([x.flat, y.flat, z.flat])

def get_train_single(num=100, multi=False, noise_only=False, close=False):
	if multi:
		filename = '3d_gauss_train_multi' +'.h5'
	else:
		filename = '3d_gauss_train' +'.h5'
	f = file(filename, mode='w')
	f.close()
	print "Creating Training Samples..."
	mu_range_spectral = [-0.7, 0.7]
	sigma_range_spectral = [0.01, 0.1]
	mu_range_xy = [-0.6, 0.6]
	sigma_range_xy = [0.2, 0.8]

	mu_spectral = numpy.random.uniform(mu_range_spectral[0], mu_range_spectral[1], size=(num,1))
	mu_xy = numpy.random.uniform(mu_range_xy[0], mu_range_xy[1], size=(num, 2))
	mu = numpy.column_stack((mu_spectral, mu_xy))

	sigma_spectral = numpy.random.uniform(sigma_range_spectral[0], sigma_range_spectral[1], size=(num,1))
	sigma_xy = numpy.random.uniform(sigma_range_xy[0], sigma_range_xy[1], size=(num, 2))
	sigma = numpy.column_stack((sigma_spectral, sigma_xy))

	mu2 = numpy.zeros(num)
	sigma2 = numpy.zeros(num)

	if multi:
		if not close:
			mu_spectral = numpy.random.uniform(mu_range_spectral[0], mu_range_spectral[1], size=(num,1))
			mu_xy = numpy.random.uniform(mu_range_xy[0], mu_range_xy[1], size=(num, 2))
			mu2 = numpy.column_stack((mu_spectral, mu_xy))
		else:
			#mu_spectral = numpy.random.uniform(mu_range_spectral[0], mu_range_spectral[1], size=(num,1))
			mu_xy = numpy.random.uniform(mu_range_xy[0], mu_range_xy[1], size=(num, 2))
			mu2 = numpy.column_stack((mu_spectral+sigma_spectral*numpy.random.uniform(3, 4, size=(num,1)), mu_xy))

		sigma_spectral = numpy.random.uniform(sigma_range_spectral[0], sigma_range_spectral[1], size=(num,1))
		sigma_xy = numpy.random.uniform(sigma_range_xy[0], sigma_range_xy[1], size=(num, 2))
		sigma2 = numpy.column_stack((sigma_spectral, sigma_xy))

	counter = 0
	out = []
	for mu, sigma, mu2, sigma2, ind in zip(mu, sigma, mu2, sigma2, range(num)):
		z = grab_single(mu, sigma, mu2, sigma2, ind, multi, filename, noise_only=noise_only)
		counter+=1
		out.append(z)
		print str(counter) + ' of ' + str(num) + ' samples completed \r',
		sys.stdout.flush()
	#numpy.save(filename, numpy.array(out))
	return out

def grab_single(mu, sigma, mu2, sigma2, ind, multi=False, filename=False, noise_only=False):
	covariance = np.diag(sigma**2)
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)

	if multi:
		covariance2 = np.diag(sigma2**2)
		z2 = multivariate_normal.pdf(xy, mean=mu2, cov=covariance2)
		z = z+z2
		tx = '3d_gauss_train_multi/test'
	else:
		tx = '3d_gauss_train/test'

	z = z.reshape(x.shape)
	if noise_only:
		z=z*0.
	else:
		z = z*(1/numpy.max(z))
	z = add_noise(z, max_noise=0.4)

	#f = file(filename, mode='a')
	#numpy.save(f, z)
	#f.close()
	#fits.writeto(tx+ str(ind) +'.fits', data=z, header=None, overwrite=True)
	return z

def add_noise(z, max_noise=0.4):
	mn = numpy.random.uniform(0.1, max_noise, size=1)
	for (i,j), value in np.ndenumerate(z[0]):
		noise=np.random.uniform(-mn[0],mn[0],len(z[:,i,j]))
		z[:,i,j] = z[:,i,j] + noise
	return z
	
out1 = get_train_single(num=40000, noise_only=False)
# Add some noise-only samples to single component class
out11 = get_train_single(num=4000, noise_only=True) 
out2 = get_train_single(num=40000, multi=True, noise_only=False)
# Add some closely separated velocity samples to multi component class
out22 = get_train_single(num=4000, multi=True, noise_only=False, close=True)
out1.extend(out11)
out2.extend(out22)
out1.extend(out2)
with h5py.File('training.h5', 'w') as hf:
	hf.create_dataset('data', data=numpy.array(out1))
	hf.close()
with h5py.File('labels.h5', 'w') as hf:
	d=numpy.append(numpy.zeros(len(out2)), numpy.ones(len(out2)))
	hf.create_dataset('data', data=d)
	hf.close()
del out1
del out2
del out11 
del out22

out1 = get_train_single(num=10000, noise_only=False)
out11 = get_train_single(num=1000, noise_only=True)
out2 = get_train_single(num=10000, multi=True, noise_only=False)
out22 = get_train_single(num=1000, multi=True, noise_only=False, close=True)
out1.extend(out11)
out2.extend(out22)
out1.extend(out2)
with h5py.File('testing.h5', 'w') as hf:
	hf.create_dataset('data', data=numpy.array(out1))
	hf.close()
with h5py.File('test_labels.h5', 'w') as hf:
	d=numpy.append(numpy.zeros(len(out2)), numpy.ones(len(out2)))
	hf.create_dataset('data', data=d)
	hf.close()

#data =  np.dstack((x,y,z))
#data = fits.getdata('MonR2_NH3_11_all_rebase3.fits')
#for (i,j), value in np.ndenumerate(data[0]):
#	data[:,i,j] = numpy.ones(numpy.shape(data[:,i,j]))
#header = fits.getheader('MonR2_NH3_11_all_rebase3.fits')

#beam = radio_beam.Beam(major=32*u.arcsec, minor=32*u.arcsec, pa=0*u.deg)
#cube_km = cube_km_1.convolve_to(beam)

