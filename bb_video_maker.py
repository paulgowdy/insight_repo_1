from PIL import Image
from glob import glob
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy.ma as ma
import csv
import pickle
import re 
import cv2


from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score, confusion_matrix

import xgboost
from xgboost import XGBClassifier, XGBRegressor


def full_pipeline_apply(image_array, high_res_shape):

	'''
	takes in an ultrasound image as a numpy array

	segments it
	- uses multiple combinations of segment # and beta value
	- 6 segment #'s, 4 beta values, gives 24 heatmaps in total

	turn segments into segment stats!

	applies a previousley trained regressor to the segments
	- regressor needs to have been trained on that N_Seg and beta value!
	- regressor outputs its prediciton for the % masked
	- these %'s, or probabilities, could be used to generate a heatmap

	** Could grab the heatmaps here and feed them into a CNN!

	each heatmap (of the 24) is shown to a second regressor 
	- each of these regressors was trained on only that segment #, beta value combo
	- these regressors output a probability that the heatmap is suspicious

	each heatmap (of the 24) is shown to a classifier
	- each of these classifiers was trained on only that segment #, beta value combo
	- these classifiers output 0 (normal) or 1 (suspicious)

	all 24 probabilities  and 24 classifier outputs are then shown to one final classifier
	- which simply outputs 0 (normal) or 1 (suspicious)

	if the output is 1 (suspicious), go back and draw a bounding box on the original image
	- the box should contain the hottest region
	- need to set the size parameters!

	return the final classifier result, the average heatmap for plotting, and the bounding box coordinates if there are any
	'''

	segments = [35, 40, 45, 50, 55, 60]
	betas = [5, 7, 8, 10] 

	# Full collection of segment/beta combination segmentation outputs
	label_collection = multi_segment(image_array, segments, betas)

	segment_regressors_fn = 'models/segment_models/segment_regressors_multiparameter.pickle'

	with open(segment_regressors_fn, 'rb') as handle:

		segment_regressors = pickle.load(handle)

	segment_classification_collector = []

	# Loop over the segmentations
	for label in label_collection:

		segment_stats = segment_stats_collector_test(image_array, label[2], label[0], label[1])


		regressor_row = segment_regressors[label_collection.index(label)]

		# print(regressor_row[0], regressor_row[1], label[0], label[1])
		# NOTE: This is badly controlled, basically relying on the fact that I made these two lists in the same order
		# Should implement a better look-up between the label (nseg and beta) and the corresponding regressor!

		regressor = regressor_row[2]

		heatmap = heatmap_from_seg_stats_regressor_labels(segment_stats, regressor, label[2])

		heatmap_classifier_fn = 'models/heatmap_models/heatmap_classifier_' + str(label[0]) + '_' + str(label[1]) + '.pickle'

		with open(heatmap_classifier_fn, 'rb') as h:

			heatmap_classifier = pickle.load(h)

		heatmap_features = heatmap_featurizer_test(heatmap)

		heatmap_features = np.asarray([heatmap_features])

		#print(heatmap_features.shape)

		heatmap_class_prediction = heatmap_classifier.predict(heatmap_features)

		segment_classification_collector.append(heatmap_class_prediction[0])

		# Build average heatmap
		if label_collection.index(label) == 0:

			average_heatmap = np.zeros(high_res_shape)

		z = sp.misc.imresize(heatmap, high_res_shape)/255.0

		z = sp.ndimage.gaussian_filter(z, sigma=5)

		average_heatmap += z

		#print(heatmap.shape)

		# plt.figure()

		# plt.subplot(1,2,1)
		# plt.imshow(image_array, cmap=plt.cm.gray)
		# plt.axis('off')

		# plt.subplot(1,2,2)
		# plt.imshow(heatmap, cmap = plt.cm.jet, vmin=0, vmax=0.6)
		# plt.axis('off')

		# plt.show()

	average_heatmap /= len(label_collection)

	segment_classification_collector = np.asarray(segment_classification_collector)

	final_classification = sp.stats.mode(segment_classification_collector)

	return final_classification[0][0], average_heatmap

def heatmap_featurizer_test(heatmap, thresh = [0.2, 0.3, 0.4, 0.5, 0.6]):

	l_to_return = []

	flat_heatmap = heatmap.flatten()

	#l_to_return.append(label)

	l_to_return.append(np.sum(flat_heatmap))
	l_to_return.append(np.mean(flat_heatmap))
	l_to_return.append(np.median(flat_heatmap))
	l_to_return.append(np.amax(flat_heatmap))
	l_to_return.append(np.amin(flat_heatmap))
	l_to_return.append(np.amax(flat_heatmap) - np.amin(flat_heatmap))

	l_to_return.append(np.percentile(flat_heatmap, 50))
	l_to_return.append(np.percentile(flat_heatmap, 75))
	l_to_return.append(np.percentile(flat_heatmap, 85))
	l_to_return.append(np.percentile(flat_heatmap, 90))
	l_to_return.append(np.percentile(flat_heatmap, 95))

	'''
	s['std'] = np.std(cr_m2)
	s['mad'] = mad(cr_m2)
	s['skew'] = sp.stats.skew(z)
	s['kurtosis'] = sp.stats.kurtosis(z)
	'''

	l_to_return.append(np.std(flat_heatmap))
	l_to_return.append(mad(flat_heatmap))
	l_to_return.append(sp.stats.skew(flat_heatmap))
	l_to_return.append(sp.stats.kurtosis(flat_heatmap))

	flat_heatmap.sort()

	z = flat_heatmap[::-1]

	l_to_return.append(np.sum(z[:int(len(z)/2.0)]))
	l_to_return.append(np.sum(z[:int(len(z)/5.0)]))
	l_to_return.append(np.sum(z[:int(len(z)/10.0)]))
	l_to_return.append(np.sum(z[:int(len(z)/20.0)]))

	l_to_return.append(np.sum(flat_heatmap[np.where(flat_heatmap > thresh[0])]))
	l_to_return.append(np.sum(flat_heatmap[np.where(flat_heatmap > thresh[1])]))
	l_to_return.append(np.sum(flat_heatmap[np.where(flat_heatmap > thresh[2])]))
	l_to_return.append(np.sum(flat_heatmap[np.where(flat_heatmap > thresh[3])]))
	l_to_return.append(np.sum(flat_heatmap[np.where(flat_heatmap > thresh[4])]))

	mid_200 = crop_center(heatmap,200,200)
	mid_100 = crop_center(heatmap,100,100)
	mid_50 = crop_center(heatmap,50,50)
	mid_10 = crop_center(heatmap,10,10)

	mid_200 = np.sum(mid_200.flatten())
	mid_100 = np.sum(mid_100.flatten())
	mid_50 = np.sum(mid_50.flatten())
	mid_10 = np.sum(mid_10.flatten())

	l_to_return.append(mid_200)
	l_to_return.append(mid_100)
	l_to_return.append(mid_50)
	l_to_return.append(mid_10)
	

	return l_to_return

def heatmap_from_seg_stats_regressor_labels(seg_stats, regressor, lbls):

	l_to_predict = []

	for s in seg_stats[1:]:

		l_to_predict.append(segment_stats_row_into_list_test(s))

	l_to_predict = np.asarray(l_to_predict)

	#print(l_to_predict.shape)

	real_predictions = list(np.clip(regressor.predict(l_to_predict),0,1))

	real_predictions.insert(0,0)

	z = np.zeros_like(lbls, dtype=np.float)

	for r in range(lbls.shape[0]):

		for c in range(lbls.shape[1]):

			z[r,c] = real_predictions[lbls[r,c]]

	return z

def segment_stats_row_into_list_test(segment_stats_row):

	l_to_return = []

	l_to_return.append(segment_stats_row['segment_id'])	
	l_to_return.append(segment_stats_row['centroid_x'])
	l_to_return.append(segment_stats_row['centroid_y'])
	l_to_return.append(segment_stats_row['total_area'])
	l_to_return.append(segment_stats_row['bb_x_dim'])
	l_to_return.append(segment_stats_row['bb_y_dim'])
	l_to_return.append(segment_stats_row['bb_x_div_total_area'])
	l_to_return.append(segment_stats_row['bb_y_div_total_area'])
	l_to_return.append(segment_stats_row['bb_x_div_bb_y'])
	l_to_return.append(segment_stats_row['total_pixel_intensity'])
	l_to_return.append(segment_stats_row['mean_pixel_intensity'])
	l_to_return.append(segment_stats_row['min_pix'])
	l_to_return.append(segment_stats_row['max_pix'])
	l_to_return.append(segment_stats_row['bb_x_div_total_area'])
	l_to_return.append(segment_stats_row['bb_x_div_total_area'])
	l_to_return.append(segment_stats_row['bb_x_div_total_area'])
	l_to_return.append(segment_stats_row['pix_range'])
	l_to_return.append(segment_stats_row['std'])
	l_to_return.append(segment_stats_row['mad'])
	l_to_return.append(segment_stats_row['skew'])
	l_to_return.append(segment_stats_row['kurtosis'])
	l_to_return.append(segment_stats_row['n_borders'])

	#l_to_return.append(segment_stats_row['min_pix'])
	#l_to_return.append(segment_stats_row['min_pix'])

	l_to_return.extend(segment_stats_row['seg_mean_differences'])

	l_to_return.extend(segment_stats_row['seg_mean_diff_if_bordering'])

	l_to_return.append(segment_stats_row['mean_border_diff'])

	l_to_return.append(segment_stats_row['mean_border_abs_diff'])

	l_to_return.append(segment_stats_row['tot_edge_pix_intensity_edges'])
	l_to_return.append(segment_stats_row['tot_edge_pix_intensity_nonedges'])
	l_to_return.append(segment_stats_row['tot_edge_pix_intensity_both'])
	l_to_return.append(segment_stats_row['mean_edge_pix_intensity_edges'])
	l_to_return.append(segment_stats_row['mean_edge_pix_intensity_nonedges'])
	l_to_return.append(segment_stats_row['mean_edge_pix_intensity_both'])

	l_to_return.append(segment_stats_row['edge_count'])
	
	return l_to_return

def multi_segment(image_array, segments, betas):

	'''
	Loop over various segment/beta parameters pairs
	segment the input image for each
	collect the label_arrays and return them
	'''

	label_arrays = []

	for n_seg in segments:

		for beta in betas:

			label_array = spectral_segmentation(image_array, beta, n_seg)

			label_arrays.append([n_seg, beta, label_array])

	return label_arrays

def spectral_segmentation(image_a, bet, N_REGIONS = 20):

	# Convert the image into a graph with the value of the gradient on the edges
	graph = image.img_to_graph(image_a)

	# Take a decreasing function of the gradient: an exponential
	# The smaller beta is, the more independent the segmentation is of the
	# actual image. For beta=1, the segmentation is approximately a voronoi
	beta = bet
	eps = 1e-6
	graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

	# Apply spectral clustering
	reg_range = np.arange(N_REGIONS)

	assign_labels = 'discretize'

	#print("Segmenting  B:", beta, "N_Segments:", N_REGIONS)

	labels = spectral_clustering(graph, n_clusters=N_REGIONS, random_state = 1, assign_labels=assign_labels)

	# Reorder labels for consistant coloring while plotting
	new_label_ordering = []

	for n in labels:

		if n not in new_label_ordering:

			new_label_ordering.append(n)

	for j in range(len(labels)):

		current_label = labels[j]
		new_label = reg_range[new_label_ordering.index(current_label)]

		labels[j] = new_label

	labels = labels.reshape(image_a.shape)

	return labels			

def image_preproc(img_filename, mask = 0, resize = 0.1, size = (400, 400)):

	img = Image.open(img_filename).convert('L')

	img.thumbnail(size, Image.ANTIALIAS)

	img_arr = np.asarray(img)

	img_arr = sp.misc.imresize(img_arr, resize) / 255.

	if not mask:

		# Forgot why I decided to filter the image - play with this...
		img_arr = sp.ndimage.filters.median_filter(img_arr,(3,3))

	return img_arr

def segment_stats_collector_test(img_arr, labels, n_segments, beta):

	'''
	segment stats: a list of dictionaries
	one dictionary for each segment containing the following descriptive stats (in this order)

	.segment #
	.centroid x-coord
	.centroid y-coord
	.total area (pixels)
	.max x dim of bounding box
	.max y dim of bounding box
	.(max x dim)/total area
	.(max y dim)/total area
	.(max x dim)/(max y dim)
	.total pixel intensity
	.mean pixel intensity
	.min pixel intensity
	.max pixel intensity
	.pixel intensity range

	.standard deviation of pixels
	.mean absolute deviation of pixels
	.scipy.stats.skew
	.scipy.stats.kurtosis

	.# of edges

	.MASK PIXELS contained in the segment
	.(mask pixels)/(total pixels)

	.[one for each other segment]
	.for each other segment, 1 if you border it, 0 if you dont
	.the difference in mean pixel intensity between current segment and each other segment
	.product of the two above

	.mean difference with bordering segments

	EDGES
	.total edge-pix intensity along edges
	mean edge-pix intensity along edges
	.total edge-pix intensity in region (non edges)
	mean edge-pix intensity in region (non edges)
	.total edge-pix intensity in region (edges and non edges)
	mean edge-pix intensity in region (edges and non edges)

	---

	Not implemented yet

	histogram of gradients from within region
	gradients along edge (identify edge pixels)
	longest actual x and y dim, not bounding box
	how central the region is in the image as a %, so like centroid x,y div the image length/width...


	'''
	#if img_arr.shape != mask.shape:

	#	mask = sp.misc.imresize(mask, img_arr.shape)

	segment_stats = []

	for n in range(n_segments):

		segment_stats.append({
			'segment_id': n,
			'n_segments': n_segments,
			'beta': beta,
			'centroid_x': 0.0,
			'centroid_y': 0.0,
			'total_area': 0,
			'bb_x_dim': 0,
			'bb_y_dim': 0,
			'bb_x_div_total_area': 0.0,
			'bb_y_div_total_area': 0.0,
			'bb_x_div_bb_y': 0.0,
			'total_pixel_intensity': 0,
			'mean_pixel_intensity': 0.0,
			'min_pix': 0,
			'max_pix': 0,
			'pix_range': 0,
			'std': 0.0,
			'mad': 0.0,
			'skew': 0.0,
			'kurtosis': 0.0,
			'bordering_segs_list': [],
			'n_borders': 0,
			#'mask_count': 0,
			#'frac_masked': 0.0,
			'one_hot_borders': [],
			'seg_mean_differences': [],
			'seg_mean_diff_if_bordering': [],
			'mean_border_diff': 0.0,
			'mean_border_abs_diff': 0.0,

			'tot_edge_pix_intensity_edges': 0,
			'tot_edge_pix_intensity_nonedges': 0,
			'tot_edge_pix_intensity_both': 0,
			'mean_edge_pix_intensity_edges': 0.0,
			'mean_edge_pix_intensity_nonedges': 0.0,
			'mean_edge_pix_intensity_both': 0.0,

			'edge_count': 0
			}) 


	centroids = ndimage.measurements.center_of_mass(labels,labels,range(n_segments))

	seg_edge_img = np.zeros_like(img_arr)

	edge_img = edge_viewer(img_arr)

	# Loop over every pixel
	for r in range(img_arr.shape[0]):

		for c in range(img_arr.shape[1]):

			current_label = labels[r][c]

			bordering = []

			for ri in [-1,0,1]:

				for ci in [-1,0,1]:

					try:

						potential_border = labels[r + ri][c + ci]

						if potential_border != current_label and potential_border not in bordering:

							bordering.append(potential_border)



						if current_label != labels[r + ri][c + ci]:

							seg_edge_img[r,c] = 1

							segment_stats[current_label]['edge_count'] += 1


							#segment_stats[current_label]['tot_edge_pix_intensity_edges'] += edge_img[r,c]

					except:

						pass

			if seg_edge_img[r,c] == 1:

				segment_stats[current_label]['tot_edge_pix_intensity_edges'] += edge_img[r,c]

			else:

				segment_stats[current_label]['tot_edge_pix_intensity_nonedges'] += edge_img[r,c]


			

			# total pixels in the region
			segment_stats[current_label]['total_area'] += 1

			# total mask pixels in the region
			

			#if mask[r,c] > 0:

				#print(current_label)
				
			#	segment_stats[current_label]['mask_count'] += 1

			# total pixel value in the region
			segment_stats[current_label]['total_pixel_intensity'] += 255 * img_arr[r][c]

			# List of all neighboring segments
			for b in bordering:

				if b not in segment_stats[current_label]['bordering_segs_list']:

					segment_stats[current_label]['bordering_segs_list'].append(b)

	

	# Loop over segments
	for s in segment_stats:

		p = segment_stats.index(s)

		s['tot_edge_pix_intensity_both'] = s['tot_edge_pix_intensity_nonedges'] + s['tot_edge_pix_intensity_edges']

		try:
			s['mean_edge_pix_intensity_edges'] = s['tot_edge_pix_intensity_edges']/float(s['edge_count'])
		except:
			s['mean_edge_pix_intensity_edges'] = 0

		try:
			s['mean_edge_pix_intensity_both'] = s['tot_edge_pix_intensity_both']/float(s['total_area'])
		except:
			s['mean_edge_pix_intensity_both'] = 0

		try:
			s['mean_edge_pix_intensity_nonedges'] = s['tot_edge_pix_intensity_nonedges']/float(s['total_area'] - s['edge_count'])
		except:
			s['mean_edge_pix_intensity_nonedges'] = 0

		# Number of borders
		s['n_borders'] = len(s['bordering_segs_list'])

		# Isolate segment
		segment_mask = ma.masked_not_equal(labels,p)
		m2 = np.multiply(img_arr, segment_mask/float(p))
		cr_m2 = 255 * nan_cropper(m2)

		# remove all masked pixels and flatten
		z = cr_m2.compressed()

		# Measures of variance
		s['std'] = np.std(cr_m2)
		s['mad'] = mad(cr_m2)
		s['skew'] = sp.stats.skew(z)
		s['kurtosis'] = sp.stats.kurtosis(z)

		# Bounding box stats
		s['bb_x_dim'] = cr_m2.shape[1]
		s['bb_y_dim'] = cr_m2.shape[0]

		# Segment min and max
		s['min_pix'] = np.amin(cr_m2)
		s['max_pix'] = np.amax(cr_m2)
		s['pix_range'] = s['max_pix'] - s['min_pix']

		# BB Dimension secondary stats
		if s['total_area'] == 0:
			s['total_area'] = 1

		s['bb_x_div_total_area'] = s['bb_x_dim']/float(s['total_area'])
		s['bb_y_div_total_area'] = s['bb_y_dim']/float(s['total_area'])
		s['bb_x_div_bb_y'] = s['bb_x_dim']/float(s['bb_y_dim'])

		# Centroid x and y
		cent = centroids[p]
		s['centroid_x'] = cent[1]
		s['centroid_y'] = cent[0]

		# Mean pixel intensity
		s['mean_pixel_intensity'] = s['total_pixel_intensity']/float(s['total_area'])

		# Masked pixel fraction
		#s['frac_masked'] = s['mask_count']/float(s['total_area'])

	for s in segment_stats:

		for j in range(n_segments):

			if j in s['bordering_segs_list']:

				s['one_hot_borders'].append(1)

			else:

				s['one_hot_borders'].append(0)

			seg_mean_diff = s['mean_pixel_intensity'] - segment_stats[j]['mean_pixel_intensity']

			s['seg_mean_differences'].append(seg_mean_diff)

			s['seg_mean_diff_if_bordering'].append(s['seg_mean_differences'][-1] * s['one_hot_borders'][-1])

	for s in segment_stats:

		m = np.sum(s['seg_mean_diff_if_bordering'])

		s['mean_border_diff'] = m/float(s['n_borders'])

		n = np.sum(np.abs(s['seg_mean_diff_if_bordering']))

		s['mean_border_abs_diff'] = n/float(s['n_borders'])

	return segment_stats

def mask_perc_check(stats):

	z = []

	for s in stats:

		z.append(s['frac_masked'])

	z.sort()

	return z[::-1]

def is_float(val):
        try:
            float(val)
        except ValueError:
            return False
        else:
            return True

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def nan_cropper(a):

	nans = np.isnan(a)
	nancols = np.all(nans, axis=0) # 10 booleans, True where col is all NAN
	nanrows = np.all(nans, axis=1) # 15 booleans

	firstcol = nancols.argmin() # 5, the first index where not NAN
	firstrow = nanrows.argmin() # 7

	lastcol = len(nancols) - nancols[::-1].argmin() # 8, last index where not NAN
	lastrow = len(nanrows) - nanrows[::-1].argmin() # 10

	return a[firstrow:lastrow,firstcol:lastcol]

def edge_viewer(im_arr):

	dx = ndimage.sobel(im_arr, 0)  # horizontal derivative
	dy = ndimage.sobel(im_arr, 1)  # vertical derivative
	mag = np.hypot(dx, dy)  # magnitude
	mag *= 255.0 / np.max(mag)  # normalize (Q&D)

	return mag

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

#image_fn = 'Data/class_test/lesion/C4_Frame (126).jpg'

image_pre = [14, 28, 29, 31]

for img_p in image_pre:

	#img_pre = 'C8_Frame'

	#save_prefix = 'Data/Unlabeled/frame_vids/' + img_pre + '/'

	segmentation_images = glob('Data/Unlabeled/Frames/C' + str(img_p) + '_Frame *.jpg')
	#segmentation_images = glob('Data/class_test/lesion/' + img_pre + '*.jpg')

	segmentation_images = sorted_nicely(segmentation_images)

	for i in range(len(segmentation_images)):

		image_fn = segmentation_images[i]
		print(image_fn)

		image_arr = image_preproc(image_fn, 0)
		image_arr_hr = Image.open(image_fn).convert('L')
		image_arr_hr = np.asarray(image_arr_hr)

		image_arr_hr_c = cv2.cvtColor(image_arr_hr, cv2.COLOR_GRAY2RGB)

		predicted_class, ave_heatmap = full_pipeline_apply(image_arr, image_arr_hr.shape)

		if predicted_class == 1.0:

			heat_mask = cv2.inRange(ave_heatmap, 0.6, 1)

			cnts = cv2.findContours(heat_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

			c = max(cnts, key=cv2.contourArea)

			(x, y, w, h) = cv2.boundingRect(c)

			#print(x,y,w,h)		

			cv2.rectangle(image_arr_hr_c, (x - int(w * 0.2), y - int(h * 0.2)), (x + int(w * 1.2), y + int(h * 1.2)), (255, 0, 0), 3)

		#print(predicted_class)

		plt.figure()
		plt.subplot(1,2,1)
		plt.imshow(image_arr_hr_c)#, cmap=plt.cm.gray)
		plt.axis('off')

		plt.subplot(1,2,2)
		#plt.imshow(ave_heatmap, cmap = plt.cm.gnuplot2, vmin=0, vmax=1)
		plt.imshow(ave_heatmap, cmap = plt.cm.jet, vmin=0, vmax=1)
		plt.axis('off')

		imsave_fn = 'bb_videos/c' + str(img_p) + '_bb/c' + str(img_p) + '_' + str(i) + '.png'
		plt.savefig(imsave_fn, bbox_inches='tight')
		#plt.show()





