import skimage
from astropy.io import fits
from keras.models import load_model
import numpy 

img_rows=10
img_cols = 10
img_depth = 500

def test_data(f='CygX_N_13CO_conv_test_smooth_clip.fits', c=1):
	# c is the class of the test data (0=single, 1=multi)
	data = fits.getdata(f)
	header = fits.getheader(f)
	print data.shape
	# Create a 2D array to place ouput predictions
	out_arr = data[0].copy()
	out_arr[:]=numpy.nan
	
	window_shape = [data.shape[0], 10,10]
	windows = skimage.util.view_as_windows(data, window_shape, 1)
	print windows[0].shape
	wshape = windows[0].shape
	#fits.writeto('test.fits', data=windows[0][0][0], overwrite=True)

	# convert class vectors to binary class matrices
	windows = windows[0]
	windows = windows.reshape(windows.shape[0]*windows.shape[1], windows.shape[2], windows.shape[3], windows.shape[4])
	X_val_new = windows.reshape(windows.shape[0], img_rows, img_cols*img_depth)
	y_val_new = numpy.zeros(windows.shape[0])+c

	count = 0
	for i in X_val_new:
		X_val_new[count] = i*(1/numpy.max(i))

	# load model
	new_model = load_model("model_3layer_40000.h5")
	print "Loaded model from disk"

	scores = new_model.evaluate(X_val_new, y_val_new, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))
	
	# Make prediction on each pixel and output as 2D fits image
	predictions = new_model.predict(X_val_new, verbose=0)
	# Reshape to get back 2D structure
	predictions = predictions.reshape(wshape[0], wshape[1])
	out_arr[4:4+predictions.shape[0], 4:4+predictions.shape[1]]=predictions
	# Format 3D header for 2D data
	del header['NAXIS3']
	del header['LBOUND3']
	#del header['OBS3']
	del header['CRPIX3']
	del header['CDELT3']
	del header['CUNIT3']
	del header['CTYPE3']
	del header['CRVAL3']
	fits.writeto(f.split('.fits')[0]+'_pred.fits', data=out_arr, header=header, overwrite=True)
	

#test_data(f='B18_HC5N_conv_test_smooth_clip.fits', c=0)
#test_data(f='CygX_N_13CO_conv_test_smooth_clip.fits', c=1)
test_data(f='W3Main_C18O_conv_test_smooth_clip.fits', c=0)
