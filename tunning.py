from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import time

def tunner1():
	### Test to find the best parameters
	color_spaces = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
	orients = [9, 11, 13]  # HOG orientations
	pix_per_cells = [8, 16] # HOG pixels per cell
	cell_per_block = 2 # HOG cells per block
	hog_channels = [0, 1, 2, "ALL"]
	spatial_sizes = [(16, 16), (32, 32)] # Spatial binning dimensions
	hist_binss = [16, 32]    # Number of histogram bins
	spatial_feat = True # Spatial features on or off
	hist_feat = True # Histogram features on or off
	hog_feat = True # HOG features on or off
	y_start_stop = [None, None] # Min and max in y to search in slide_window()

	orient = orients[0]
	pix_per_cell = pix_per_cells[0]
	spatial_size = spatial_sizes[0]
	hist_bins = hist_binss[0]
	for color_space in color_spaces:
		for hog_channel in hog_channels:
			car_features = extract_features(veh_images, color_space=color_space, 
									spatial_size=spatial_size, hist_bins=hist_bins, 
									orient=orient, pix_per_cell=pix_per_cell, 
									cell_per_block=cell_per_block, 
									hog_channel=hog_channel, spatial_feat=spatial_feat, 
									hist_feat=hist_feat, hog_feat=hog_feat)
			notcar_features = extract_features(nonveh_images, color_space=color_space, 
									spatial_size=spatial_size, hist_bins=hist_bins, 
									orient=orient, pix_per_cell=pix_per_cell, 
									cell_per_block=cell_per_block, 
									hog_channel=hog_channel, spatial_feat=spatial_feat, 
									hist_feat=hist_feat, hog_feat=hog_feat)

			# Create an array stack of feature vectors
			X = np.vstack((car_features, notcar_features)).astype(np.float64)

			# Define the labels vector
			y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

			# Split up data into randomized training and test sets
			rand_state = np.random.randint(0, 100)
			X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=0.2, random_state=rand_state)

			# Fit a per-column scaler
			X_scaler = StandardScaler().fit(X_train)
			# Apply the scaler to X
			X_train = X_scaler.transform(X_train)
			X_test = X_scaler.transform(X_test)

			print('Using:',color_space,'color space',orient,'orientations',pix_per_cell,
				'pixels per cell and', cell_per_block,'cells per block', 
				  hog_channel, 'hog channel', spatial_size, 'spatial size', )
			print('Feature vector length:', len(X_train[0]))
			# Use a linear SVC 
			svc = LinearSVC()
			# Check the training time for the SVC
			t=time.time()
			svc.fit(X_train, y_train)
			t2 = time.time()
			print(round(t2-t, 2), 'Seconds to train SVC...')
			# Check the score of the SVC
			print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
			# Check the prediction time for a single sample
			t=time.time()
			
def tunner2():
	color_spaces = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
	orients = [9, 11]  # HOG orientations
	pix_per_cells = [8, 16] # HOG pixels per cell
	cell_per_block = 2 # HOG cells per block
	hog_channels = [0, 1, 2, "ALL"]
	spatial_sizes = [(16, 16), (32, 32)] # Spatial binning dimensions
	hist_binss = [16, 32]    # Number of histogram bins
	spatial_feat = True # Spatial features on or off
	hist_feat = True # Histogram features on or off
	hog_feat = True # HOG features on or off
	y_start_stop = [None, None] # Min and max in y to search in slide_window()

	#pairs = [('HSV', 'ALL'), ('YUV', 1), ('YUV', 'ALL'), ('YCrCb', 'ALL'), ('LUV', 'ALL'), ('HSV', 2)]
	pairs = [('YUV', 'ALL')]


	for pair in pairs:
		color_space = pair[0]
		hog_channel = pair[1]
		
		for orient in orients:
			for pix_per_cell in pix_per_cells:
				for spatial_size in spatial_sizes:
					for hist_bins in hist_binss:
					
						car_features = extract_features(veh_images, color_space=color_space, 
												spatial_size=spatial_size, hist_bins=hist_bins, 
												orient=orient, pix_per_cell=pix_per_cell, 
												cell_per_block=cell_per_block, 
												hog_channel=hog_channel, spatial_feat=spatial_feat, 
												hist_feat=hist_feat, hog_feat=hog_feat)
						notcar_features = extract_features(nonveh_images, color_space=color_space, 
												spatial_size=spatial_size, hist_bins=hist_bins, 
												orient=orient, pix_per_cell=pix_per_cell, 
												cell_per_block=cell_per_block, 
												hog_channel=hog_channel, spatial_feat=spatial_feat, 
												hist_feat=hist_feat, hog_feat=hog_feat)

						# Create an array stack of feature vectors
						X = np.vstack((car_features, notcar_features)).astype(np.float64)

						# Define the labels vector
						y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

						# Split up data into randomized training and test sets
						rand_state = np.random.randint(0, 100)
						X_train, X_test, y_train, y_test = train_test_split(
							X, y, test_size=0.2, random_state=rand_state)

						# Fit a per-column scaler
						X_scaler = StandardScaler().fit(X_train)
						# Apply the scaler to X
						X_train = X_scaler.transform(X_train)
						X_test = X_scaler.transform(X_test)

						print('Using:',color_space,'color space',orient,'orientations',pix_per_cell,
							'pixels per cell and', cell_per_block,'cells per block', 
							  hog_channel, 'hog channel', spatial_size, 'spatial size', 
							 hist_bins, 'hist bins')
						print('Feature vector length:', len(X_train[0]))
						# Use a linear SVC 
						svc = LinearSVC()
						# Check the training time for the SVC
						t=time.time()
						svc.fit(X_train, y_train)
						t2 = time.time()
						print(round(t2-t, 2), 'Seconds to train SVC...')
						# Check the score of the SVC
						print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
						# Check the prediction time for a single sample
						t=time.time()